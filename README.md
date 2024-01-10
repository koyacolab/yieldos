# yieldos
Project for Predicting Crop Yields at the Regional Level in China.

This project is dedicated to predicting crop yields at the regional level using the Temporal Fusion Transformer neural network from the pytorch-forecasting framework [pytorch.forecasting](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html#).

The project utilizes the following data sources:

I. Weather data - FLDAS: Famine Early Warning Systems Network (FEWS NET) Land Data Assimilation System from Google Datasets [FLDAS dataset](https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001).

II. Satellite MODIS data:

   - MOD09A1.061 Terra Surface Reflectance 8-Day Global 500m [MOD09A1 dataset](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09A1).
   - MOD11A2.061 Terra Land Surface Temperature and Emissivity 8-Day Global 1km [MOD11A2 dataset](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A2).
   - MOD17A2H.061: Terra Gross Primary Productivity 8-Day Global 500m [MOD17A2H dataset](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD17A2H).
   - MOD16A2.061: Terra Net Evapotranspiration 8-Day Global 500m [MOD16A2 dataset](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD16A2).

Both Weather and MODIS data have been downloaded and interpolated to a 500-meter grid.

For all regions, Weather and MODIS data are processed into time series histograms similar to - [pycrop-yield-prediction repository](https://github.com/gabrieltseng/pycrop-yield-prediction).

To predict crop yield, the Temporal Fusion Transformer utilizes these time series histograms in the `A0.py` script, which is trained with the parameters specified in the `A0.sh` script.

***************************************************************************************************
Example, for wheat (T/Ha) for 2017 year:

![image (1)](https://github.com/koyacolab/yieldos/assets/115004547/c879907e-9697-499b-b09d-1cbe9cf2c64c)

Corn (T/Ha) example for 2018 year for China counties (Target/Predicted):

![Corn_Ch](https://github.com/koyacolab/yieldos/assets/115004547/a80bb5d1-a10b-43f3-805a-f596482ab265)
***************************************************************************************************

N.B.!!!

Given the significant variation in size among Chinese regions, the preparation of time series histograms is crucial for accurate crop yield prediction. Neural networks are renowned for their ability to perform unsupervised feature extraction. 

As a result, XGBoost techniques have been proposed for the preparation of time series histograms, and the corresponding code is available in the XGBoost folder.

***************************************************************************************************
