// Copyright 2021-2023 Protocol Labs
// SPDX-License-Identifier: Apache-2.0, MIT
use fvm::call_manager::CallManager;
use fvm::gas::Gas;
use fvm::kernel::prelude::*;
use fvm::kernel::Result;
use fvm::kernel::{
    ActorOps, CryptoOps, DebugOps, EventOps, IpldBlockOps, MessageOps, NetworkOps, RandomnessOps,
    SelfOps, SendOps, SyscallHandler, UpgradeOps,
};
use fvm::syscalls::Linker;
use fvm::DefaultKernel;
use fvm_ipld_encoding::RawBytes;
use fvm_shared::clock::ChainEpoch;
use fvm_shared::randomness::RANDOMNESS_LENGTH;
use fvm_shared::sys::out::network::NetworkContext;
use fvm_shared::sys::out::vm::MessageContext;
use fvm_shared::{address::Address, econ::TokenAmount, ActorID, MethodNum};
use smartcore::ensemble::random_forest_classifier::RandomForestClassifier;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::{
    LinearRegression, LinearRegressionParameters, LinearRegressionSolverName,
};
use smartcore::linear::logistic_regression::LogisticRegression;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::neighbors::knn_classifier::KNNClassifier;
use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use smartcore::tree::decision_tree_regressor::DecisionTreeRegressor;
use std::cmp;

use ambassador::Delegate;
use cid::Cid;

pub trait MLSyscallKernel: Kernel {
    fn train_linear_regression_syscall(&self, data: &[u8], label: &[u8]) -> Result<RawBytes>;
    fn predict_linear_regression_syscall(&self, model: &[u8], test_data: &[u8])
        -> Result<RawBytes>;
    fn train_logistic_regression_syscall(&self, data: &[u8], labels: &[u8]) -> Result<RawBytes>;
    fn predict_logistic_regression_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes>;
    fn train_knn_regression_syscall(&self, data: &[u8], labels: &[u8]) -> Result<RawBytes>;
    fn predict_knn_regression_syscall(&self, model: &[u8], test_data: &[u8]) -> Result<RawBytes>;
    fn train_knn_classification_syscall(&self, data: &[u8], labels: &[u8]) -> Result<RawBytes>;
    fn predict_knn_classification_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes>;
    fn train_decision_tree_regression_syscall(
        &self,
        data: &[u8],
        labels: &[u8],
    ) -> Result<RawBytes>;
    fn predict_decision_tree_regression_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes>;
    fn train_decision_tree_classification_syscall(
        &self,
        data: &[u8],
        labels: &[u8],
    ) -> Result<RawBytes>;
    fn predict_decision_tree_classification_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes>;
    fn train_random_forest_regression_syscall(
        &self,
        data: &[u8],
        labels: &[u8],
    ) -> Result<RawBytes>;
    fn predict_random_forest_regression_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes>;
    fn train_random_forest_classification_syscall(
        &self,
        data: &[u8],
        labels: &[u8],
    ) -> Result<RawBytes>;
    fn predict_random_forest_classification_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes>;
}

// our mlsyscall kernel extends the filecoin kernel
#[derive(Delegate)]
#[delegate(IpldBlockOps, where = "C: CallManager")]
#[delegate(ActorOps, where = "C: CallManager")]
#[delegate(CryptoOps, where = "C: CallManager")]
#[delegate(DebugOps, where = "C: CallManager")]
#[delegate(EventOps, where = "C: CallManager")]
#[delegate(MessageOps, where = "C: CallManager")]
#[delegate(NetworkOps, where = "C: CallManager")]
#[delegate(RandomnessOps, where = "C: CallManager")]
#[delegate(SelfOps, where = "C: CallManager")]
#[delegate(SendOps<K>, generics = "K", where = "K: MLSyscallKernel")]
#[delegate(UpgradeOps<K>, generics = "K", where = "K: MLSyscallKernel")]
pub struct MLSyscallKernelImpl<C>(pub DefaultKernel<C>);

impl<C> MLSyscallKernel for MLSyscallKernelImpl<C>
where
    C: CallManager,
    MLSyscallKernelImpl<C>: Kernel,
{
    fn train_linear_regression_syscall(&self, data: &[u8], labels: &[u8]) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(data)),
        )
        .unwrap();

        let deserialized_labels: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(labels)),
        )
        .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let input_y: Vec<f64> = deserialized_labels
            .iter()
            .map(|&x| x as f64 / divisor as f64)
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let lir: LinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>> = LinearRegression::fit(
            &x,
            &input_y,
            LinearRegressionParameters {
                solver: LinearRegressionSolverName::QR,
            },
        )
        .unwrap();

        let model_ser = fvm_ipld_encoding::RawBytes::serialize(lir).unwrap();

        Ok(model_ser)
    }

    fn predict_linear_regression_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(test_data)),
        )
        .unwrap();
        let serialized_model: Vec<u8> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(model)),
        )
        .unwrap();
        let deserialized_model: LinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            fvm_ipld_encoding::RawBytes::deserialize(&fvm_ipld_encoding::RawBytes::new(
                serialized_model,
            ))
            .unwrap();

        let divisor: i64 = 100;
        let multiplier: f64 = 100.0;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let prediction = deserialized_model.predict(&x).unwrap();

        let result: Vec<i64> = prediction
            .iter()
            .map(|&x| (x * multiplier) as i64)
            .collect();

        let ser_result_raw = fvm_ipld_encoding::RawBytes::serialize(result).unwrap();
        Ok(ser_result_raw)
    }

    fn train_logistic_regression_syscall(&self, data: &[u8], labels: &[u8]) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(data)),
        )
        .unwrap();

        let deserialized_labels: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(labels)),
        )
        .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let input_y: Vec<i64> = deserialized_labels;

        let x = DenseMatrix::from_2d_vec(&input_x);

        let lir: LogisticRegression<f64, i64, DenseMatrix<f64>, Vec<i64>> =
            LogisticRegression::fit(&x, &input_y, Default::default()).unwrap();

        let model_ser = fvm_ipld_encoding::RawBytes::serialize(lir).unwrap();

        Ok(model_ser)
    }

    fn predict_logistic_regression_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(test_data)),
        )
        .unwrap();
        let serialized_model: Vec<u8> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(model)),
        )
        .unwrap();
        let deserialized_model: LogisticRegression<f64, i64, DenseMatrix<f64>, Vec<i64>> =
            fvm_ipld_encoding::RawBytes::deserialize(&fvm_ipld_encoding::RawBytes::new(
                serialized_model,
            ))
            .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let prediction = deserialized_model.predict(&x).unwrap();

        let result: Vec<i64> = prediction;

        let ser_result_raw = fvm_ipld_encoding::RawBytes::serialize(result).unwrap();
        Ok(ser_result_raw)
    }

    fn train_knn_regression_syscall(&self, data: &[u8], labels: &[u8]) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(data)),
        )
        .unwrap();

        let deserialized_labels: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(labels)),
        )
        .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let input_y: Vec<f64> = deserialized_labels
            .iter()
            .map(|&x| x as f64 / divisor as f64)
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let knn: KNNRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>, Euclidian<f64>> =
            KNNRegressor::fit(&x, &input_y, Default::default()).unwrap();

        let model_ser = fvm_ipld_encoding::RawBytes::serialize(knn).unwrap();

        Ok(model_ser)
    }

    fn predict_knn_regression_syscall(&self, model: &[u8], test_data: &[u8]) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(test_data)),
        )
        .unwrap();
        let serialized_model: Vec<u8> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(model)),
        )
        .unwrap();
        let deserialized_model: KNNRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>, Euclidian<f64>> =
            fvm_ipld_encoding::RawBytes::deserialize(&fvm_ipld_encoding::RawBytes::new(
                serialized_model,
            ))
            .unwrap();

        let divisor: i64 = 100;
        let multiplier: f64 = 100.0;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let prediction = deserialized_model.predict(&x).unwrap();

        let result: Vec<i64> = prediction
            .iter()
            .map(|&x| (x * multiplier) as i64)
            .collect();

        let ser_result_raw = fvm_ipld_encoding::RawBytes::serialize(result).unwrap();

        Ok(ser_result_raw)
    }

    fn train_knn_classification_syscall(&self, data: &[u8], labels: &[u8]) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(data)),
        )
        .unwrap();

        let deserialized_labels: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(labels)),
        )
        .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let input_y: Vec<i64> = deserialized_labels;

        let x = DenseMatrix::from_2d_vec(&input_x);

        let knn: KNNClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>, Euclidian<f64>> =
            KNNClassifier::fit(&x, &input_y, Default::default()).unwrap();

        let model_ser = fvm_ipld_encoding::RawBytes::serialize(knn).unwrap();

        Ok(model_ser)
    }

    fn predict_knn_classification_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(test_data)),
        )
        .unwrap();
        let serialized_model: Vec<u8> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(model)),
        )
        .unwrap();
        let deserialized_model: KNNClassifier<
            f64,
            i64,
            DenseMatrix<f64>,
            Vec<i64>,
            Euclidian<f64>,
        > = fvm_ipld_encoding::RawBytes::deserialize(&fvm_ipld_encoding::RawBytes::new(
            serialized_model,
        ))
        .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let prediction = deserialized_model.predict(&x).unwrap();

        let result: Vec<i64> = prediction;

        let ser_result_raw = fvm_ipld_encoding::RawBytes::serialize(result).unwrap();
        Ok(ser_result_raw)
    }

    fn train_decision_tree_regression_syscall(
        &self,
        data: &[u8],
        labels: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(data)),
        )
        .unwrap();

        let deserialized_labels: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(labels)),
        )
        .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let input_y: Vec<f64> = deserialized_labels
            .iter()
            .map(|&x| x as f64 / divisor as f64)
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let decision_tree: DecisionTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            DecisionTreeRegressor::fit(&x, &input_y, Default::default()).unwrap();

        let model_ser = fvm_ipld_encoding::RawBytes::serialize(decision_tree).unwrap();

        Ok(model_ser)
    }

    fn predict_decision_tree_regression_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(test_data)),
        )
        .unwrap();
        let serialized_model: Vec<u8> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(model)),
        )
        .unwrap();
        let deserialized_model: DecisionTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            fvm_ipld_encoding::RawBytes::deserialize(&fvm_ipld_encoding::RawBytes::new(
                serialized_model,
            ))
            .unwrap();

        let divisor: i64 = 100;
        let multiplier: f64 = 100.0;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let prediction = deserialized_model.predict(&x).unwrap();

        let result: Vec<i64> = prediction
            .iter()
            .map(|&x| (x * multiplier) as i64)
            .collect();

        let ser_result_raw = fvm_ipld_encoding::RawBytes::serialize(result).unwrap();
        Ok(ser_result_raw)
    }

    fn train_decision_tree_classification_syscall(
        &self,
        data: &[u8],
        labels: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(data)),
        )
        .unwrap();

        let deserialized_labels: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(labels)),
        )
        .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let input_y: Vec<i64> = deserialized_labels;

        let x = DenseMatrix::from_2d_vec(&input_x);

        let decision_tree: DecisionTreeClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>> =
            DecisionTreeClassifier::fit(&x, &input_y, Default::default()).unwrap();

        let model_ser = fvm_ipld_encoding::RawBytes::serialize(decision_tree).unwrap();

        Ok(model_ser)
    }

    fn predict_decision_tree_classification_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(test_data)),
        )
        .unwrap();
        let serialized_model: Vec<u8> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(model)),
        )
        .unwrap();
        let deserialized_model: DecisionTreeClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>> =
            fvm_ipld_encoding::RawBytes::deserialize(&fvm_ipld_encoding::RawBytes::new(
                serialized_model,
            ))
            .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let prediction = deserialized_model.predict(&x).unwrap();

        let result: Vec<i64> = prediction;

        let ser_result_raw = fvm_ipld_encoding::RawBytes::serialize(result).unwrap();
        Ok(ser_result_raw)
    }

    fn train_random_forest_regression_syscall(
        &self,
        data: &[u8],
        labels: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(data)),
        )
        .unwrap();

        let deserialized_labels: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(labels)),
        )
        .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let input_y: Vec<f64> = deserialized_labels
            .iter()
            .map(|&x| x as f64 / divisor as f64)
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let random_forest = RandomForestRegressor::fit(&x, &input_y, Default::default()).unwrap();

        let model_ser = fvm_ipld_encoding::RawBytes::serialize(random_forest).unwrap();

        Ok(model_ser)
    }

    fn predict_random_forest_regression_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(test_data)),
        )
        .unwrap();
        let serialized_model: Vec<u8> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(model)),
        )
        .unwrap();
        let deserialized_model: RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>> =
            fvm_ipld_encoding::RawBytes::deserialize(&fvm_ipld_encoding::RawBytes::new(
                serialized_model,
            ))
            .unwrap();

        let divisor: i64 = 100;
        let multiplier: f64 = 100.0;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let prediction = deserialized_model.predict(&x).unwrap();

        let result: Vec<i64> = prediction
            .iter()
            .map(|&x| (x * multiplier) as i64)
            .collect();

        let ser_result_raw = fvm_ipld_encoding::RawBytes::serialize(result).unwrap();
        Ok(ser_result_raw)
    }

    fn train_random_forest_classification_syscall(
        &self,
        data: &[u8],
        labels: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(data)),
        )
        .unwrap();

        let deserialized_labels: Vec<i64> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(labels)),
        )
        .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let input_y: Vec<i64> = deserialized_labels;

        let x = DenseMatrix::from_2d_vec(&input_x);

        let random_forest: RandomForestClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>> =
            RandomForestClassifier::fit(&x, &input_y, Default::default()).unwrap();

        let model_ser = fvm_ipld_encoding::RawBytes::serialize(random_forest).unwrap();

        Ok(model_ser)
    }

    fn predict_random_forest_classification_syscall(
        &self,
        model: &[u8],
        test_data: &[u8],
    ) -> Result<RawBytes> {
        let deserialized_data: Vec<Vec<i64>> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(test_data)),
        )
        .unwrap();
        let serialized_model: Vec<u8> = fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(model)),
        )
        .unwrap();
        let deserialized_model: RandomForestClassifier<f64, i64, DenseMatrix<f64>, Vec<i64>> =
            fvm_ipld_encoding::RawBytes::deserialize(&fvm_ipld_encoding::RawBytes::new(
                serialized_model,
            ))
            .unwrap();

        let divisor: i64 = 100;

        // Check to prevent division by zero
        let input_x: Vec<Vec<f64>> = deserialized_data
            .iter()
            .map(|inner_vec| {
                inner_vec
                    .iter()
                    .map(|&x| x as f64 / divisor as f64)
                    .collect()
            })
            .collect();

        let x = DenseMatrix::from_2d_vec(&input_x);

        let prediction = deserialized_model.predict(&x).unwrap();

        let result: Vec<i64> = prediction;

        let ser_result_raw = fvm_ipld_encoding::RawBytes::serialize(result).unwrap();
        Ok(ser_result_raw)
    }
}

impl<C> Kernel for MLSyscallKernelImpl<C>
where
    C: CallManager,
{
    type CallManager = C;
    type Limiter = <DefaultKernel<C> as Kernel>::Limiter;

    fn into_inner(self) -> (Self::CallManager, BlockRegistry)
    where
        Self: Sized,
    {
        self.0.into_inner()
    }

    fn new(
        mgr: C,
        blocks: BlockRegistry,
        caller: ActorID,
        actor_id: ActorID,
        method: MethodNum,
        value_received: TokenAmount,
        read_only: bool,
    ) -> Self {
        MLSyscallKernelImpl(DefaultKernel::new(
            mgr,
            blocks,
            caller,
            actor_id,
            method,
            value_received,
            read_only,
        ))
    }

    fn machine(&self) -> &<Self::CallManager as CallManager>::Machine {
        self.0.machine()
    }

    fn limiter_mut(&mut self) -> &mut Self::Limiter {
        self.0.limiter_mut()
    }

    fn gas_available(&self) -> Gas {
        self.0.gas_available()
    }

    fn charge_gas(&self, name: &str, compute: Gas) -> Result<GasTimer> {
        self.0.charge_gas(name, compute)
    }
}

impl<K> SyscallHandler<K> for MLSyscallKernelImpl<K::CallManager>
where
    K: MLSyscallKernel
        + ActorOps
        + SendOps
        + UpgradeOps
        + IpldBlockOps
        + CryptoOps
        + DebugOps
        + EventOps
        + MessageOps
        + NetworkOps
        + RandomnessOps
        + SelfOps,
{
    fn link_syscalls(linker: &mut Linker<K>) -> anyhow::Result<()> {
        DefaultKernel::<K::CallManager>::link_syscalls(linker)?;

        linker.link_syscall(
            "mlsyscall_kernel",
            "train_linear_regression_syscall",
            train_linear_regression_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "predict_linear_regression_syscall",
            predict_linear_regression_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "train_logistic_regression_syscall",
            train_logistic_regression_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "predict_logistic_regression_syscall",
            predict_logistic_regression_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "train_knn_regression_syscall",
            train_knn_regression_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "predict_knn_regression_syscall",
            predict_knn_regression_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "train_knn_classification_syscall",
            train_knn_classification_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "predict_knn_classification_syscall",
            predict_knn_classification_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "train_decision_tree_regression_syscall",
            train_decision_tree_regression_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "predict_decision_tree_regression_syscall",
            predict_decision_tree_regression_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "train_decision_tree_classification_syscall",
            train_decision_tree_classification_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "predict_decision_tree_classification_syscall",
            predict_decision_tree_classification_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "train_random_forest_regression_syscall",
            train_random_forest_regression_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "predict_random_forest_regression_syscall",
            predict_random_forest_regression_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "train_random_forest_classification_syscall",
            train_random_forest_classification_syscall,
        )?;
        linker.link_syscall(
            "mlsyscall_kernel",
            "predict_random_forest_classification_syscall",
            predict_random_forest_classification_syscall,
        )?;

        Ok(())
    }
}

pub fn train_linear_regression_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    conv_offset: u32,
    conv_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let conv_array = context.memory.try_slice(conv_offset, conv_length)?;
    let ser_result_raw = context
        .kernel
        .train_linear_regression_syscall(array, conv_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn predict_linear_regression_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    model_offset: u32,
    model_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let model_array = context
        .memory
        .try_slice(model_offset as u32, model_length as u32)?;

    let data_array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let ser_result_raw = context
        .kernel
        .predict_linear_regression_syscall(model_array, data_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn train_logistic_regression_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    conv_offset: u32,
    conv_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let conv_array = context.memory.try_slice(conv_offset, conv_length)?;
    let ser_result_raw = context
        .kernel
        .train_logistic_regression_syscall(array, conv_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn predict_logistic_regression_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    model_offset: u32,
    model_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let model_array = context
        .memory
        .try_slice(model_offset as u32, model_length as u32)?;

    let data_array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let ser_result_raw = context
        .kernel
        .predict_logistic_regression_syscall(model_array, data_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn train_knn_regression_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    conv_offset: u32,
    conv_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let conv_array = context.memory.try_slice(conv_offset, conv_length)?;
    let ser_result_raw = context
        .kernel
        .train_knn_regression_syscall(array, conv_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn predict_knn_regression_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    model_offset: u32,
    model_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let model_array = context
        .memory
        .try_slice(model_offset as u32, model_length as u32)?;

    let data_array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let ser_result_raw = context
        .kernel
        .predict_knn_regression_syscall(model_array, data_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn train_knn_classification_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    conv_offset: u32,
    conv_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let conv_array = context.memory.try_slice(conv_offset, conv_length)?;
    let ser_result_raw = context
        .kernel
        .train_knn_classification_syscall(array, conv_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn predict_knn_classification_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    model_offset: u32,
    model_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let model_array = context
        .memory
        .try_slice(model_offset as u32, model_length as u32)?;

    let data_array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let ser_result_raw = context
        .kernel
        .predict_knn_classification_syscall(model_array, data_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn train_decision_tree_regression_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    conv_offset: u32,
    conv_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let conv_array = context.memory.try_slice(conv_offset, conv_length)?;
    let ser_result_raw = context
        .kernel
        .train_decision_tree_regression_syscall(array, conv_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn predict_decision_tree_regression_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    model_offset: u32,
    model_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let model_array = context
        .memory
        .try_slice(model_offset as u32, model_length as u32)?;

    let data_array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let ser_result_raw = context
        .kernel
        .predict_decision_tree_regression_syscall(model_array, data_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn train_decision_tree_classification_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    conv_offset: u32,
    conv_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let conv_array = context.memory.try_slice(conv_offset, conv_length)?;
    let ser_result_raw = context
        .kernel
        .train_decision_tree_classification_syscall(array, conv_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn predict_decision_tree_classification_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    model_offset: u32,
    model_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let model_array = context
        .memory
        .try_slice(model_offset as u32, model_length as u32)?;

    let data_array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let ser_result_raw = context
        .kernel
        .predict_decision_tree_classification_syscall(model_array, data_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn train_random_forest_regression_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    conv_offset: u32,
    conv_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let conv_array = context.memory.try_slice(conv_offset, conv_length)?;
    let ser_result_raw = context
        .kernel
        .train_random_forest_regression_syscall(array, conv_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn predict_random_forest_regression_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    model_offset: u32,
    model_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let model_array = context
        .memory
        .try_slice(model_offset as u32, model_length as u32)?;

    let data_array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let ser_result_raw = context
        .kernel
        .predict_random_forest_regression_syscall(model_array, data_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn train_random_forest_classification_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    conv_offset: u32,
    conv_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let conv_array = context.memory.try_slice(conv_offset, conv_length)?;
    let ser_result_raw = context
        .kernel
        .train_random_forest_classification_syscall(array, conv_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}

pub fn predict_random_forest_classification_syscall(
    context: fvm::syscalls::Context<'_, impl MLSyscallKernel>,
    data_offset: u32,
    data_length: u32,
    output_offset: u32,
    output_length: u32,
    model_offset: u32,
    model_length: u32,
) -> Result<u32> {
    // Check the digest bounds first so we don't do any work if they're incorrect.
    context.memory.check_bounds(output_offset, output_length)?;

    let model_array = context
        .memory
        .try_slice(model_offset as u32, model_length as u32)?;

    let data_array = context
        .memory
        .try_slice(data_offset as u32, data_length as u32)?;

    let ser_result_raw = context
        .kernel
        .predict_random_forest_classification_syscall(model_array, data_array)?;

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context
        .memory
        .try_slice_mut(output_offset, output_length)
        .unwrap();
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}
