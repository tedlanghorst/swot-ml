import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from tqdm import tqdm
import rasterio.features
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import transform
import pyproj
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import dask.config
from dask.distributed import Client, as_completed
from dask_jobqueue import SLURMCluster
from sklearn.cluster import KMeans
from data import BasinDataLake

ERA5_DIR = Path("/nas/cee-ice/data/ERA5-Land/")


def create_spatial_batches(gdf, batch_size):
    """
    Groups subbasins into spatially coherent batches using K-Means clustering.
    
    Returns a list of batch dictionaries, each containing the IDs and
    a single bounding box for all geometries in that batch.
    """
    n_batches = int(np.ceil(len(gdf)/batch_size))
        
    # Get centroids for clustering. 
    # Projecting mostly to avoid the geographic coordinate warnings, 
    # as these centroids do not need to be very accurate.
    gdf_proj = gdf.to_crs("EPSG:3857")
    centroids = np.array([list(p.coords)[0] for p in gdf_proj.geometry.centroid])
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_batches, random_state=42, n_init=10)
    gdf['cluster'] = kmeans.fit_predict(centroids)
    
    batches = []
    for i in range(n_batches):
        batch_gdf = gdf[gdf['cluster'] == i]
        if batch_gdf.empty:
            continue
        
        # Get all COMIDs in this cluster
        batch_ids = batch_gdf.index.tolist()
        
        # Calculate a single, unified bounding box for the entire batch
        # Add a 0.1 degree padding to ensure all grid cells are included
        min_lon, min_lat, max_lon, max_lat = batch_gdf.total_bounds
        padding = 0.1
        batch_bounds = (
            np.floor((min_lon - padding) * 10) / 10,
            np.floor((min_lat - padding) * 10) / 10,
            np.ceil((max_lon + padding) * 10) / 10,
            np.ceil((max_lat + padding) * 10) / 10,
        )
        
        batches.append({
            "ids": batch_ids,
            "geoms": batch_gdf.geometry,
            "bounds": batch_bounds
        })
        
    return batches


def open_monthly_files(year, month):
    """Loads and merges the original monthly ERA5 files."""
    fps = [
        ERA5_DIR / "accumulated" / f"{year}_{month:02d}.nc",
        ERA5_DIR / "average" / f"{year}_{month:02d}.nc"
    ]
    datasets = [xr.open_dataset(fp, engine="h5netcdf") for fp in fps]
    ds = xr.merge(datasets, compat="override").drop_vars(['expver','number'], errors='ignore')
    return ds


def subset_ds_by_bounds(ds, bounds):
    """Subset dataset by spatial bounds."""
    min_lon, min_lat, max_lon, max_lat = bounds
    return ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))


def get_weight_matrix(ds_subset, watershed_geom):
    """Calculate weight matrix for a single watershed."""
    lat, lon = ds_subset['latitude'].values, ds_subset['longitude'].values
    lat_res, lon_res = 0.1, 0.1
    
    watershed_proj = watershed_geom.to_crs(epsg=3857)
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    
    coverage_fractions = np.zeros((len(lat), len(lon)))
    
    for i, lat_val in enumerate(lat):
        for j, lon_val in enumerate(lon):
            pixel_box = box(lon_val - lon_res/2, lat_val - lat_res/2,
                            lon_val + lon_res/2, lat_val + lat_res/2)
            pixel_box_proj = transform(transformer.transform, pixel_box)
            
            intersection_area = watershed_proj.intersection(pixel_box_proj).area
            pixel_area = pixel_box_proj.area
            coverage_fractions[i, j] = (intersection_area / pixel_area).iloc[0] if pixel_area > 0 else 0
    
    return xr.DataArray(
        coverage_fractions,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon}
    )


def get_weight_matrix_fast(ds_subset, watershed_geom, scale_factor=10):
    """
    Calculates fractional coverage using super-sampling rasterization.
    
    Args:
        ds_subset: xarray Dataset/DataArray with latitude/longitude coords
        watershed_geom: GeoDataFrame containing the watershed polygon
        scale_factor: Factor to upsample for precision (10 = 1/100th precision)
    """
    # 1. Extract coordinates and resolution
    # Assumes 1D coordinates for lat/lon
    lat = ds_subset['latitude'].values
    lon = ds_subset['longitude'].values
    
    # Calculate resolution (assuming uniform grid)
    # ERA5 lat is usually descending, so d_lat might be negative
    d_lat = lat[1] - lat[0]
    d_lon = lon[1] - lon[0]
    
    # 2. Define Coarse Dimensions
    height = len(lat)
    width = len(lon)
    
    # 3. Define Fine Dimensions (Super-sampling)
    fine_height = height * scale_factor
    fine_width = width * scale_factor
    
    # 4. Create Affine Transform for the FINE grid
    # We shift the origin to the top-left corner of the first pixel
    # Note: We subtract/add half a pixel to move from center to corner
    west = lon[0] - d_lon / 2
    north = lat[0] - d_lat / 2
    
    # The fine resolution is the coarse resolution divided by the scale factor
    fine_transform = from_origin(
        west, 
        north, 
        d_lon / scale_factor, 
        d_lat / scale_factor # rasterio handles negative dy correctly
    )
    
    # 5. Rasterize onto the fine grid (Binary 0/1)
    # We use the projected geometry if possible, but rasterio expects 
    # the geometry coords to match the transform coords (Lat/Lon here).
    # Ensure watershed_geom is EPSG:4326 before passing in.
    if watershed_geom.crs != "EPSG:4326":
        watershed_geom = watershed_geom.to_crs("EPSG:4326")
        
    mask_fine = rasterio.features.rasterize(
        shapes=[(watershed_geom.geometry.iloc[0], 1)],
        out_shape=(fine_height, fine_width),
        transform=fine_transform,
        fill=0,
        all_touched=False,  # We rely on high-res center-points for coverage
        dtype=np.float32
    )
    
    # 6. Aggregate (downsample) to coarse grid
    # Reshape to (Coarse_H, Factor, Coarse_W, Factor)
    # Then take the mean over the two 'Factor' axes
    mask_coarse = mask_fine.reshape(
        height, scale_factor, width, scale_factor
    ).mean(axis=(1, 3))
    
    # 7. Return as DataArray matching input ds
    return xr.DataArray(
        mask_coarse,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="weights"
    )


def get_results_df(ds_sub, weights):
    """Calculate weighted statistics."""
    dims = ["latitude", "longitude"]
    ds_sub_w = ds_sub.weighted(weights)
    
    mean_df = (ds_sub_w.mean(dim=dims).to_dataframe()
               .rename(lambda n: n+"_mean", axis=1))
    var_df = (ds_sub_w.var(dim=dims).to_dataframe()
               .rename(lambda n: n+"_var", axis=1))
    
    return pd.concat([mean_df, var_df], axis=1)


def process_spatial_batch(spatial_batch: dict, start_date, end_date):
    """
    Processes a spatially coherent batch of COMIDs.
    Loads regional data once per month and applies it to all COMIDs in the batch.
    """
    batch_ids = spatial_batch['ids']
    batch_geoms = gpd.GeoDataFrame(spatial_batch['geoms'])
    batch_bounds = spatial_batch['bounds']

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Pre-calculate weight matrices for all watersheds in the batch
    # This requires a sample dataset to get the grid right
    print(f"Pre-calculating weight matrices for batch...")
    sample_ds = open_monthly_files(start_date.year, start_date.month)
    sample_ds_sub = subset_ds_by_bounds(sample_ds, batch_bounds)
    
    weights_map = {}
    for comid, row in batch_geoms.iterrows():
        watershed = gpd.GeoDataFrame([row], crs=batch_geoms.crs)
        weights_map[comid] = get_weight_matrix(sample_ds_sub, watershed)
    sample_ds.close()
    print(f"Weight matrices calculated.")

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    
    # This dictionary will hold the time-series data for each COMID
    comid_results = {comid: [] for comid in batch_ids}

    # Outer loop is TIME
    for date in date_range:
        # Load the full monthly file and subset to the BATCH's bounds ONCE
        ds_full = open_monthly_files(date.year, date.month)
        ds_region = subset_ds_by_bounds(ds_full, batch_bounds)
        
        # Inner loop is COMID (processing from in-memory data)
        for comid in batch_ids:
            weights = weights_map[comid]
            
            # Ensure weights align with the regional data subset
            weights_aligned = weights.reindex_like(ds_region, method='nearest', tolerance=0.01)
            
            df_month = get_results_df(ds_region, weights_aligned)
            comid_results[comid].append(df_month)
        
        ds_full.close()
        ds_region.close()

    cat_dfs = {}
    for comid, df_list in comid_results.items():
        cat_dfs[comid] = pd.concat(df_list, axis=0)

    return cat_dfs


def process_and_write_batch(basin_id, batch_dict, start_date, end_date, save_dir):
    # Process the batch to get the dataframes in memory (on the worker)
    subbasin_df_dict = process_spatial_batch(batch_dict, start_date, end_date)
    
    # Initialize the store connection
    store = BasinDataLake(save_dir)
    store.write_dynamic(basin_id, 'era5', subbasin_df_dict, mode='append')
    
    # Just return number of subbasins we finished
    return len(subbasin_df_dict)



if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin-file", required=True,
                        help="Path to subbasin file directory")
    parser.add_argument("--save-dir", required=True,
                        help="Location of the deltalake dataset")
    parser.add_argument("--n-workers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-date", default="1980-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    
    args = parser.parse_args()
    print("\n========== Program Arguments ==========")
    for k, v in vars(args).items():
        print(f"{k:>12}: {v}")
    print("=======================================\n")

    print ("Reading subbasin file ...")
    subbasins = gpd.read_parquet(args.basin_file)

    # Place Mississippi first. 
    target = 'USGS-07374000'
    subbasins = pd.concat([
        subbasins[subbasins["outlet_id"] == target], 
        subbasins[subbasins["outlet_id"] != target]
    ])

    print("Preparing subbasin batches ...")
    # Query the store metadata and determine which still need processing
    store = BasinDataLake(args.save_dir)
    
    # status = store.get_processing_status(source='era5')
    # processed_subs = [] if status.empty else status.index.get_level_values('subbasin').unique()
    # to_process = subbasins[~subbasins.index.isin(processed_subs)]
    to_process = subbasins

    # Precalculate all the batches we need to iterate over.
    all_batches = []
    for basin_name, basin_gdf in to_process.groupby('outlet_id'):
        batches = create_spatial_batches(basin_gdf, args.batch_size)
        for b in batches:
            all_batches.append((basin_name, b))

    print("Starting dask cluster ...")
    # Increase heartbeat limits to tolerate long-running GIL tasks
    dask.config.set({
        "distributed.comm.timeouts.connect": "60s",  # Time to establish connection
        "distributed.comm.timeouts.tcp": "60s",      # Time to wait for a response
        "distributed.scheduler.worker-ttl": "120s",  # Scheduler waits this long before removing worker (Default 30s-60s)
    })
    
    cluster = SLURMCluster(
        job_name="era5-dask-worker",
        queue="ceewater_cjgleason-cpu",
        cores=1,
        processes=1,
        memory="32GB",
        walltime="7-00:00:00",
        log_directory=Path.cwd() / "_dask_workers",
        job_extra_directives=["-q long", "-A pi_cjgleason_umass_edu"],
        job_script_prologue=[
            "cd /nas/cee-water/cjgleason/ted/swot-ml",
            "source .venv/bin/activate",
        ],
    )
    
    try:
        cluster.scale(args.n_workers)
        client = Client(cluster)
        client.wait_for_workers(args.n_workers)

        # submission loop
        futures_dict = {}
        for basin_id, batch_dict in all_batches:
            future = client.submit(
                process_and_write_batch,
                basin_id,
                batch_dict,
                args.start_date,
                args.end_date,
                args.save_dir
            )
            futures_dict[future] = basin_id

        # results loop
        with tqdm(total=len(futures_dict), desc="Processing spatial batches") as pbar:
            for future in as_completed(futures_dict):
                basin_id = futures_dict[future]
                try:
                    num_processed = future.result()
                    print(f"Batch for {basin_id} completed, processed {num_processed} subbasins.")
                except Exception as e:
                    print(f"batch for {basin_id} failed: {e}")
                finally:
                    pbar.update(1)

    finally:
        if 'client' in locals() and client.status != 'closed':
            client.close()
        if 'cluster' in locals() and cluster.status != 'closed':
            cluster.close()