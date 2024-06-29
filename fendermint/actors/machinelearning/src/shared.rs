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

#[derive(FromPrimitive)]
#[repr(u64)]
pub enum Method {
    TrainLinearRegression = frc42_dispatch::method_hash!("TrainLinearRegression"),
    PredictLinearRegression = frc42_dispatch::method_hash!("PredictLinearRegression"),
    TrainLogisticRegression = frc42_dispatch::method_hash!("TrainLogisticRegression"),
    PredictLogisticRegression = frc42_dispatch::method_hash!("PredictLogisticRegression"),
}
