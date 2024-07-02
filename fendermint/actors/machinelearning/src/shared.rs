// Copyright 2021-2023 Protocol Labs
// SPDX-License-Identifier: Apache-2.0, MIT
use fvm_ipld_encoding::tuple::{Deserialize_tuple, Serialize_tuple};
use num_derive::FromPrimitive;

pub const MACHINELEARNING_ACTOR_NAME: &str = "machinelearning";

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct TrainLinearRegressionParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub labels: Vec<i64>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct PredictLinearRegressionParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub model: Vec<u8>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct TrainLogisticRegressionParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub labels: Vec<i64>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct PredictLogisticRegressionParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub model: Vec<u8>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct TrainKNNRegressionParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub labels: Vec<i64>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct PredictKNNRegressionParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub model: Vec<u8>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct TrainKNNClassificationParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub labels: Vec<i64>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct PredictKNNClassificationParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub model: Vec<u8>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct TrainDecisionTreeRegressionParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub labels: Vec<i64>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct PredictDecisionTreeRegressionParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub model: Vec<u8>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct TrainDecisionTreeClassificationParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub labels: Vec<i64>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct PredictDecisionTreeClassificationParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub model: Vec<u8>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct TrainRandomForestRegressionParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub labels: Vec<i64>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct PredictRandomForestRegressionParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub model: Vec<u8>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct TrainRandomForestClassificationParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub labels: Vec<i64>,
}

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple)]
pub struct PredictRandomForestClassificationParams {
    pub input_matrix: Vec<Vec<i64>>,
    pub model: Vec<u8>,
}

#[derive(FromPrimitive)]
#[repr(u64)]
pub enum Method {
    TrainLinearRegression = frc42_dispatch::method_hash!("TrainLinearRegression"),
    PredictLinearRegression = frc42_dispatch::method_hash!("PredictLinearRegression"),
    TrainLogisticRegression = frc42_dispatch::method_hash!("TrainLogisticRegression"),
    PredictLogisticRegression = frc42_dispatch::method_hash!("PredictLogisticRegression"),
    TrainKNNRegression = frc42_dispatch::method_hash!("TrainKNNRegression"),
    PredictKNNRegression = frc42_dispatch::method_hash!("PredictKNNRegression"),
    TrainKNNClassification = frc42_dispatch::method_hash!("TrainKNNClassification"),
    PredictKNNClassification = frc42_dispatch::method_hash!("PredictKNNClassification"),
    TrainDecisionTreeRegression = frc42_dispatch::method_hash!("TrainDecisionTreeRegression"),
    PredictDecisionTreeRegression = frc42_dispatch::method_hash!("PredictDecisionTreeRegression"),
    TrainDecisionTreeClassification =
        frc42_dispatch::method_hash!("TrainDecisionTreeClassification"),
    PredictDecisionTreeClassification =
        frc42_dispatch::method_hash!("PredictDecisionTreeClassification"),
    TrainRandomForestRegression = frc42_dispatch::method_hash!("TrainRandomForestRegression"),
    PredictRandomForestRegression = frc42_dispatch::method_hash!("PredictRandomForestRegression"),
    TrainRandomForestClassification =
        frc42_dispatch::method_hash!("TrainRandomForestClassification"),
    PredictRandomForestClassification =
        frc42_dispatch::method_hash!("PredictRandomForestClassification"),
}
