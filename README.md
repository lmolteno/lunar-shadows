# Lunar Shadows

This project is for creating maps of lunar occultations - where the moon passes in front of a star from our perspective. It is focussed on providing accurate DEM-corrections so that observing locations for
grazing occultations can be accurately chosen.

It can project the lunar limb onto the earth from the perspective of a star at a given moment, or it can project the lunar limb from a particular observer over time, allowing you to 'watch the star' move behind it.

## Mapping
The main objective is to map the lunar limb onto the earth, correcting for the terrain of the moon and earth. This is done using the dataset from Astropedia ([LRO LOLA 118m](https://astrogeology.usgs.gov/search/map/moon_lro_lola_dem_118m)), and copernicus elevation data downloaded from [OpenTopography](https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.1). Higher resolution DEM's exist but I that level of detail is unnecessary for this purpose. This is performed by running `main.py` to generate the projected limb points then `plot.py` to show them alongside data (in this case from [LINZ](data.linz.govt.nz)).
![high-res](https://github.com/user-attachments/assets/13e06d62-47dd-42d7-8269-e7f172cd942b)
![zoomed-in](https://github.com/user-attachments/assets/43aec462-0750-43ef-9cce-49b90a18d49e)

## POV
A secondary ability is calculating the view from an observer. This is done by running `observer.py`. It uses data from Astropedia for crater mapping to provide context.
![observer](https://github.com/user-attachments/assets/36e62d8c-76c9-4c66-947c-715e05e8531c)
