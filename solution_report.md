# Research Center Quality Classification — Report by Atefeh Momeni

## Overview

This project classifies research centers into quality categories using unsupervised machine learning.

The classification is based on:
- internal infrastructure
- nearby healthcare access
- facility diversity
- facility density

The workflow includes:

1. Exploratory Data Analysis (EDA)
2. Feature selection
3. Data scaling
4. K-Means clustering
5. Cluster interpretation
6. API deployment using FastAPI

The final categories are:

- Premium
- Standard
- Basic


--------------------------------------------------

## 1. Exploratory Data Analysis (EDA)

The dataset was first inspected to understand its structure and quality.

The following checks were performed:

- dataset shape
- data types
- missing values
- duplicate rows
- summary statistics

Results:

- No missing values were found
- No duplicate rows were found
- Numeric columns were already clean
- Categorical columns were identifiers or location descriptors

This means the dataset did not require data cleaning before modeling.


--------------------------------------------------

## 2. Plot Interpretation

### Histogram of `internalFacilitiesCount`

The histogram shows that research centers differ noticeably in the number of internal facilities.

Interpretation:

- Some centers have very low internal facility counts
- Some centers have medium counts
- A smaller number of centers have high counts

This suggests that internal infrastructure is one of the strongest indicators of quality.
Centers with more labs, testing units, or workstations are more likely to belong to a higher-quality tier.


### Scatter plots of hospitals and pharmacies access

The scatter plots comparing `hospitals_10km` and `pharmacies_10km` show that centers with stronger external healthcare access tend to be grouped together.

Interpretation:

- Higher hospital access usually appears together with higher pharmacy access
- Centers with low surrounding healthcare availability appear in a separate region of the plot
- This supports the idea that external healthcare access is an important part of research center quality


### Correlation heatmap interpretation

The heatmap was used to understand how numeric features relate to each other.

Key interpretations:

- `facilityDensity_10km` has a positive relationship with `hospitals_10km` and `pharmacies_10km`
- `facilityDiversity_10km` is positively related to density, meaning areas with more facilities are also usually more diverse
- `internalFacilitiesCount` is an important independent quality indicator because it reflects internal strength of the center
- `latitude` and `longitude` are not meaningful quality measures; they mainly capture geography
- Identifier fields such as `researchCenterId` and `researchCenterName` do not carry useful clustering information

Conclusion from the heatmap:

Research center quality is influenced by both:
- internal strength (`internalFacilitiesCount`)
- external ecosystem quality (`hospitals_10km`, `pharmacies_10km`, `facilityDiversity_10km`, `facilityDensity_10km`)

This supports selecting these five variables for clustering.


--------------------------------------------------

## 3. Feature Selection

The following features were selected for clustering:

- internalFacilitiesCount
- hospitals_10km
- pharmacies_10km
- facilityDiversity_10km
- facilityDensity_10km

These were selected because they directly describe the quality of a research center.

Removed features:

- researchCenterId
- researchCenterName
- city
- latitude
- longitude

Reason for removal:

- `researchCenterId` and `researchCenterName` are identifiers
- `city` is categorical and does not directly measure quality
- `latitude` and `longitude` describe geographic position, not infrastructure quality

The goal of clustering is to group centers by quality, not by name or map position.


--------------------------------------------------

## 4. Data Scaling

StandardScaler was applied before clustering.

Reason:

K-Means is distance-based.
Without scaling, features with larger numeric ranges would dominate the clustering process.

Scaling ensures that all selected variables contribute more fairly.


--------------------------------------------------

## 5. Clustering Model

K-Means clustering was used.

The number of clusters was fixed to:

k = 3

This matches the assignment requirement to classify centers into:

- Premium
- Standard
- Basic

Model settings:

- `n_clusters = 3`
- `random_state = 42`
- `n_init = 10`

Silhouette score obtained:

0.55

Interpretation:

A silhouette score around 0.55 indicates reasonably good separation between clusters for a small synthetic dataset.
This suggests that the three groups are meaningful and not random.


--------------------------------------------------

## 6. Cluster Results

After fitting K-Means, the average values of the selected features were calculated for each cluster.

The results showed a clear ordering:

### Basic cluster
Characteristics:

- lowest internalFacilitiesCount
- lowest hospital access
- lowest pharmacy access
- lowest facility density
- lowest facility diversity

Interpretation:

This cluster represents lower-quality centers with limited internal resources and weaker surrounding healthcare support.


### Standard cluster
Characteristics:

- medium internalFacilitiesCount
- medium hospital and pharmacy access
- medium density and diversity

Interpretation:

This cluster represents centers with moderate quality.
They have acceptable infrastructure and surrounding support, but not at the highest level.


### Premium cluster
Characteristics:

- highest internalFacilitiesCount
- highest hospital access
- highest pharmacy access
- highest facility density
- highest facility diversity

Interpretation:

This cluster represents the highest-quality centers.
These centers have both strong internal infrastructure and strong nearby healthcare support.


--------------------------------------------------

## 7. Final Clusters Identified

The final quality groups found in the data are:

- **Premium**: centers with the strongest internal facilities and best surrounding healthcare environment
- **Standard**: centers with medium infrastructure and moderate surrounding support
- **Basic**: centers with the weakest infrastructure and lowest surrounding healthcare access

So the final conclusion from clustering is that the dataset naturally separates into three meaningful quality tiers:

**Basic → Standard → Premium**

This ordering is consistent across all selected features.


--------------------------------------------------

## 8. Mapping Numeric Clusters to Labels

K-Means returns numeric cluster IDs only.

To make the output interpretable, cluster averages were ranked from highest to lowest.

The mapping rule was:

- highest mean feature profile → Premium
- middle mean feature profile → Standard
- lowest mean feature profile → Basic

This makes the clustering output understandable and usable in the API.


--------------------------------------------------

## 9. Why k = 3 was used

The value of k was set to 3 because the assignment explicitly defines three target tiers:

- Premium
- Standard
- Basic

Therefore, extensive hyperparameter tuning for the number of clusters was not necessary.

The silhouette score was used to confirm that k = 3 still gives acceptable clustering quality.


--------------------------------------------------

## 10. API Deployment

The trained model artifacts were saved as:

- `kmeans_model.pkl`
- `scaler.pkl`
- `cluster_mapping.json`

A FastAPI endpoint was created:

`POST /predict`

Example input:

```json
{
  "internalFacilitiesCount": 9,
  "hospitals_10km": 3,
  "pharmacies_10km": 2,
  "facilityDiversity_10km": 0.82,
  "facilityDensity_10km": 0.45
}
