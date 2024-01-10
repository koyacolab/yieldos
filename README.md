# yieldos
project for crop yield prediction for China region level

This project is dedicated to predicting crop yields at the regional level using the Temporal Fusion Transformer neural network from the pytorch-forecasting framework [pytorch.forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html#).

The project utilizes the following data sources:

I. Weather data - FLDAS: Famine Early Warning Systems Network (FEWS NET) Land Data Assimilation System from Google Datasets [FLDAS dataset](https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001).

II. Satellite MODIS data:

   - MOD09A1.061 Terra Surface Reflectance 8-Day Global 500m [MOD09A1 dataset](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1).
   - MOD11A2.061 Terra Land Surface Temperature and Emissivity 8-Day Global 1km [MOD11A2 dataset](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2).
   - MOD17A2H.061: Terra Gross Primary Productivity 8-Day Global 500m [MOD17A2H dataset](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD17A2H).
   - MOD16A2.061: Terra Net Evapotranspiration 8-Day Global 500m [MOD16A2 dataset](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2).

Both weather and MODIS data have been downloaded and interpolated to a 500-meter grid.

For all regions, weather and MODIS data are processed into time series histograms similar to - [pycrop-yield-prediction repository](https://github.com/gabrieltseng/pycrop-yield-prediction).

To predict crop yield, the Temporal Fusion Transformer utilizes these time series histograms in the `A0.py` script, which is trained with the parameters specified in the `A0.sh` script.

