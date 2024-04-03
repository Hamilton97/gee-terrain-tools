# Google Earth Engine Terrain Tools
from typing import Any

import ee
from tagee import terrainAnalysis


def guassian_filter(
    image: ee.Image,
    radius: float = 3,
    sigma: int = 2,
    units: str = None,
    normalize: bool = True,
    magnitude: float = 1.0,
):
    return image.convolve(
        ee.Kernel.gaussian(radius, sigma, units, normalize, magnitude)
    )


def perona_malik_filter(image: ee.Image, K=3.5, iterations: int = 10, method: int = 2):
    dxW = ee.Kernel.fixed(3, 3, [[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    dxE = ee.Kernel.fixed(3, 3, [[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    dyN = ee.Kernel.fixed(3, 3, [[0, 1, 0], [0, -1, 0], [0, 0, 0]])
    dyS = ee.Kernel.fixed(3, 3, [[0, 0, 0], [0, -1, 0], [0, 1, 0]])

    lamb = 0.2
    k1 = ee.Image(-1.0 / K)
    k2 = ee.Image(K).multiply(ee.Image(K))

    for _ in range(0, iterations):
        dI_W = image.convolve(dxW)
        dI_E = image.convolve(dxE)
        dI_N = image.convolve(dyN)
        dI_S = image.convolve(dyS)

        if method == 1:
            cW = dI_W.multiply(dI_W).multiply(k1).exp()
            cE = dI_E.multiply(dI_E).multiply(k1).exp()
            cN = dI_N.multiply(dI_N).multiply(k1).exp()
            cS = dI_S.multiply(dI_S).multiply(k1).exp()

            image = image.add(
                ee.Image(lamb).multiply(
                    cN.multiply(dI_N)
                    .add(cS.multiply(dI_S))
                    .add(cE.multiply(dI_E))
                    .add(cW.multiply(dI_W))
                )
            )

    if method == 2:
        cW = ee.Image(1.0).divide(ee.Image(1.0).add(dI_W.multiply(dI_W).divide(k2)))
        cE = ee.Image(1.0).divide(ee.Image(1.0).add(dI_E.multiply(dI_E).divide(k2)))
        cN = ee.Image(1.0).divide(ee.Image(1.0).add(dI_N.multiply(dI_N).divide(k2)))
        cS = ee.Image(1.0).divide(ee.Image(1.0).add(dI_S.multiply(dI_S).divide(k2)))

        image = image.add(
            ee.Image(lamb).multiply(
                cN.multiply(dI_N)
                .add(cS.multiply(dI_S))
                .add(cE.multiply(dI_E))
                .add(cW.multiply(dI_W))
            )
        )

    return image


# steps are build the rectangle from what ever aoi you are using

TERRAIN_PRODUCTS = [
    "Elevation",
    "Slope",
    "Aspect",
    "Hillshade",
    "Northness",
    "Eastness",
    "HorizontalCurvature",
    "VerticalCurvature",
    "MeanCurvature",
    "GaussianCurvature",
    "MinimalCurvature",
    "MaximalCurvature",
    "ShapeIndex",
]


def create_rectangle(geometry):
    if isinstance(geometry, ee.FeatureCollection):
        geometry = geometry.geometry()
    # create a rectangle
    coords = geometry.bounds().coordinates()

    listCoords = ee.Array.cat(coords, 1)
    xCoords = listCoords.slice(1, 0, 1)
    yCoords = listCoords.slice(1, 1, 2)

    xMin = xCoords.reduce("min", [0]).get([0, 0])
    xMax = xCoords.reduce("max", [0]).get([0, 0])
    yMin = yCoords.reduce("min", [0]).get([0, 0])
    yMax = yCoords.reduce("max", [0]).get([0, 0])

    return ee.Geometry.Rectangle(xMin, yMin, xMax, yMax)


def compute_gaussian_bands(
    dem: ee.Image, aoi: ee.Geometry, selectors: list[str] = None
) -> ee.Image:
    selectors = selectors or ["Elevation", "Slope", "GaussianCurvature"]
    dem_filtered = guassian_filter(dem)
    return compute_terrain_analysis(dem_filtered, aoi)


def compute_perona_malik_bands(dem, aoi, selectors: list[str] = None) -> ee.Image:
    selectors = selectors or [
        "HorizontalCurvature",
        "VerticalCurvature",
        "MeanCurvature",
    ]
    dem_filtered = perona_malik_filter(dem)
    return compute_terrain_analysis(dem_filtered, aoi)


def compute_terrain_analysis(dem, aoi):
    bbox = create_rectangle(aoi)
    dem = dem.clip(bbox)
    return terrainAnalysis(dem, bbox)
