# yieldos
project for crop yield prediction for China region level

This project for crop yield prediction at the region level use the Temporal Fusion Transformer neural network from the pytorch-forecasting framework 

(https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html#)

This project used:

I. Weather data - FLDAS: Famine Early Warning Systems Network (FEWS NET) Land Data Assimilation System from Google Datasets

(https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001)

II. Sattelite MODIS data:

1. MOD09A1.061 Terra Surface Reflectance 8-Day Global 500m
       
(https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1)
                
2. MOD11A2.061 Terra Land Surface Temperature and Emissivity 8-Day Global 1km
       
(https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2)
                
3. MOD17A2H.061: Terra Gross Primary Productivity 8-Day Global 500m 
       
(https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD17A2H)
                
4. MOD16A2.061: Terra Net Evapotranspiration 8-Day Global 500m
       
(https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2)


Weather and MODIS data downloaded and interpolated to 500 meters grid. 

For all regions Weather and MODIS data are processed to time series histograms like - https://github.com/gabrieltseng/pycrop-yield-prediction.

For crop yield predicting this time series histograms are utilized by the Temporal Fusion Transformer in A0.py script and trained with A0.sh parameters. 



