import numpy as np
import rasterio
import skfuzzy as fuzz
import os

### USER-MODIFIED VARIABLES ###

# Input raster
in_raster = r"dNBR_raster.tif"
# Number of clusters
n_clusters = 6
# Output folder
out_folder = r"data\Burn_Severity6"
# Filename prefix
prefix = "FCM6"

### FCM ###

# Load raster
with rasterio.open(in_raster) as src:
    arr = src.read(1).astype(float)
    profile = src.profile
    nodata = src.nodata

# Mask NoData
masked = np.ma.masked_equal(arr, nodata)
data = masked.compressed()
data_2d = np.expand_dims(data, 0)

arcpy.AddMessage("Running fuzzy c-means...")

# Run FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data_2d,
    c = n_clusters,
    m = 2.0,
    error = 0.005,
    maxiter = 1000,
    init = None
)

# Sort clusters & create hard classification
order = np.argsort(cntr.flatten())

cntr_sorted = cntr[order]
u_sorted = u[order, :]

flat_mask = masked.mask.flatten()
membership_rasters = np.zeros((n_clusters, arr.shape[0], arr.shape[1]), dtype=float)

for i in range(n_clusters):
    memb_flat = np.zeros(arr.size, dtype=float)
    memb_flat[~flat_mask] = u_sorted[i]
    membership_rasters[i] = memb_flat.reshape(arr.shape)

valid_classes = np.argmax(u_sorted, axis = 0) + 1
hard_flat = np.zeros(arr.size, dtype = "int16")
hard_flat[~flat_mask] = valid_classes
hard_flat[flat_mask] = 0

hard_raster = hard_flat.reshape(arr.shape)

# Ensure output folder exists
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# Rasterio profile updates
profile.update(dtype="float32", count=1, nodata=0)

# Save cluster rasters
for i in range(n_clusters):
    out_path = os.path.join(out_folder, f"{prefix}_cluster{i+1}.tif")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(membership_rasters[i].astype("float32"), 1)

# Save hard classification raster
hard_path = os.path.join(out_folder, f"{prefix}_hard_classification.tif")

with rasterio.open(hard_path, "w", **profile) as dst:
    dst.write(hard_raster.astype("int16"), 1)