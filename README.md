Codes for *Lakes reshape the structure of temperature-humidity extremes and compound risks in a lake-rich region*

## Project Structure

### Main scripts

- `Analyzes.py`: Main workflow for running analysis and generating figures.
- `Validation.py`: Workflow for model validation and validation figures.
- `StudyAreaPlot.py`: Plots the study area map and related geographic layers.
- `Compute_HotWet_Exposure.py`: Computes hot-wet exposure and affected population statistics.
- `HotWet_Decomposition.py`: Decomposes changes in hot-wet event probabilities.
- `Func_Additonal.py`: Additional analysis functions for regional climate impacts.
- `Func_CalIndex.py`: Calculates model skill and lake climate impact indices.
- `Func_CouplingRelationship.py`: Calculates and plots temperature-humidity coupling metrics.
- `Func_ExtremeEvent.py`: Defines, counts, analyzes, and plots extreme events.
- `Func_Manuscript.py`: Merges panels for manuscript and supplementary figures.
- `Func_ModelValidation.py`: Plots and merges seasonal model validation results.
- `Func_PressureLevel.py`: Analyzes pressure-level variables and atmospheric stability.
- `Func_RegClimImpact.py`: Analyzes and plots regional climate impacts.
- `Func_Review1.py`: Generates figures prepared for review response materials.
- `Func_SpatialMap.py`: Plots spatial maps, wind fields, and diurnal cycles.
- `Func_StationValidation.py`: Validates model output against station observations.
- `Func_WarmAdvection.py`: Calculates and plots warm advection changes.

### ToolBoxes

- `ToolBoxes/__init__.py`: Initializes the helper module package.
- `ToolBoxes/Tool_CouplingTest.py`: Statistical tools for coupling and tail dependence.
- `ToolBoxes/Tool_DataPrepare.py`: Prepares data for maps, boxplots, rose plots, and probability plots.
- `ToolBoxes/Tool_ExtremeEventMetrics.py`: Core metrics and classifiers for hot, wet, and compound events.
- `ToolBoxes/Tool_ImageToolkit.py`: Image reading, cropping, merging, overlaying, and PDF export helpers.
- `ToolBoxes/Tool_InputOutput.py`: Reads, processes, and saves CWRF, reference, NetCDF, Excel, and raster data.
- `ToolBoxes/Tool_LoadData.py`: Loads surface and pressure-level variables.
- `ToolBoxes/Tool_NCLUtils.py`: NCL-like utilities for pressure, vertical coordinates, and humidity.
- `ToolBoxes/Tool_PerformanceMetrics.py`: Model performance metrics such as RMSE, bias, correlation, and skill scores.
- `ToolBoxes/Tool_PlotAreaMap.py`: Map plotting helpers for Lambert projection, shapefiles, masks, and spatial fields.
- `ToolBoxes/Tool_PlotBar.py`: Bar and stacked-bar plotting helpers.
- `ToolBoxes/Tool_PlotBox.py`: Boxplot helpers for differences and relative contributions.
- `ToolBoxes/Tool_PlotCircularRing.py`: Circular ring plot helpers.
- `ToolBoxes/Tool_PlotColorBar.py`: Standalone colorbar plotting helpers.
- `ToolBoxes/Tool_PlotConfig.py`: Shared plot styles, variable metadata, event metadata, and path settings.
- `ToolBoxes/Tool_PlotHeatMap.py`: Heatmap plotting helpers.
- `ToolBoxes/Tool_PlotJoint.py`: Joint distribution plotting helpers.
- `ToolBoxes/Tool_PlotKDE.py`: One-dimensional KDE plotting helper.
- `ToolBoxes/Tool_PlotLine.py`: Line plot helpers for diurnal and seasonal comparisons.
- `ToolBoxes/Tool_PlotRadialHistogram.py`: Radial histogram plotting helper.
- `ToolBoxes/Tool_PlotRose.py`: Rose plot helpers for diurnal climate impact figures.
- `ToolBoxes/Tool_PlotTreemap.py`: Treemap plotting helpers.
- `ToolBoxes/Tool_SaturationVaporPressure.py`: Saturation vapor pressure and humidity-related calculations.
- `ToolBoxes/Tool_SignificanceTest.py`: Paired significance tests and grid-based significance analysis.
- `ToolBoxes/Tool_WetBulbTemperature.py`: Wet-bulb and globe temperature calculations.
- `ToolBoxes/Tool_YangtzeColorMap.py`: Color maps, labels, plotting ranges, and crop parameters used by the project.
- `ToolBoxes/Utils.py`: General utilities for smoothing, calendars, CWRF grid area, lake masks, and wind speed.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
