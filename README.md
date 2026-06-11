# Visualization Project

Link to the GitHub: https://github.com/dashkim/clouds

This project was using data from the World Data Center for Climate (WDCC) and Copernicus (ERA5). The data is not included because the files were huge. I used to 20 minute series in two dimensions, which had really good temporal-spatial resolution, but it did not have the scale I needed.

Most importantly, there was NO INVERSION in the dataset, so I had to supplement with a larger dataset from ERA5. 


## Summary
The project is a movie. I wanted to give visibiltiy to what cloud inversion was, as well as display the relationship between the most important features and how it translated to inversion.

Where to watch: output/composite
youtube: https://youtu.be/8TlPONBAZ5A?si=J4QKRl7E8c7_WIS_

In a little more depth, The beggining is the two dimensional dataset from the IEEE conference (WDCC). It visualizes cloud coverage from April 16th, 2013 for 20 minutes. 

The next section is looking at specific section of the German Alps, where we look at cloud coverage and wind velocities.

Lastly, the final section compares my ML model with the actual data, showing how my model compares to reality, alongside the change in the most valuable features from the model. 

## Data

Place files in `data/`:

| File | Source |
|------|--------|
| `3d_icon_dom*.nc` | ICON 3D DOM (vertical cloud structure) |
| `2d_lonlat*.nc` | ICON 2D lon/lat grid (cloud cover over time) |
| `2013_germany.grib` | ERA5 reanalysis for ML and overlays |

Download from [scivis2017.dkrz.de](https://scivis2017.dkrz.de/)
Request from Copernicus Climate Institute


## Scripts

- `scripts/visualize_lonlat_vtk.py` — interactive 2D viewer
- `scripts/visualize_clouds_vtk.py` — interactive 3D viewer
- `scripts/render_composite_movie.py` — build the full composite movie
- `scripts/train_inversion_regression.py` — train ridge model
- `scripts/train_inversion_cnn.py` — train CNN model

There's a lot of scripts, but lots of them are self explanitory. I started by rendering in the section as GIFS, and then because the gifs were weirdly compressed, the movie is made by a different mp4 script. 


## Important Notes: 

- The movie uses the data, but a lot of work had to be done to get them working in VTK. I was attracted to this data at first because they had great 3d stuff. However, anything meaningful was at least 200 GB large. It was not feasible to work with it on the scale I wanted. The dataset I ended up using was a 2d 20 minute interval, much smaller (8 GB)
The 3d meshes are extrapolated from the 2d file. Instead of directly using the data, I had to write a script that added more dimensionality. They are pseudo-3d slabs on terrain. The ERA5 data also isn't the same resolution, which you can see in the movie. This is evident in the cloud deck visualization, as it looks like a massive sheet rather than the smooth clouds in the initial cloud visualizations. 

- I was orignally planing on doing a lightweight for the model, as I thought it would be interesting. However, I didn't have the computational resources to train a model that large, and initial epochs had terrible accuracy. There was also limited inversions so training was unstable. I also had trouble with the data being a time series. Lastly, I thought ridge regression would be more condusive to a movie because feature relationship clarity. The legacy code is still there in case you want a peek. 


## Challenges: 

- Translating the 2d into 3d, better for visualizing
- Getting the data: Getting started took weeks to get approved. Also working with the APIs for ERA5
- Defining Inversions and sorting through noise..., loading the model's outputs as data on the mesh. 
- The CNN (ended up unused)
- Getting the smaller gifs to show up uncompressed/weird
- translating different resolutions into a comprehensive movie