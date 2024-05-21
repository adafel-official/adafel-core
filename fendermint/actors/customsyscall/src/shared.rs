// Copyright 2021-2023 Protocol Labs
// SPDX-License-Identifier: Apache-2.0, MIT

use fvm_ipld_encoding::tuple::{Deserialize_tuple, Serialize_tuple};
use num_derive::FromPrimitive;

pub const CUSTOMSYSCALL_ACTOR_NAME: &str = "customsyscall";

#[derive(Default, Debug, Serialize_tuple, Deserialize_tuple, Copy, Clone)]
pub struct InvokeParams {
    pub i1: u64,
    pub i2: u64,
    pub i3: u64,
    // pub i4: u64,
    // pub i5: u64,
    // pub i6: u64,
    // pub i7: u64,
    // pub i8: u64,
    // pub i9: u64,
    pub c1: u64,
    pub c2: u64,
    pub c3: u64,
    // pub c4: u64,
    // pub c5: u64,
    // pub c6: u64,
    // pub c7: u64,
    // pub c8: u64,
    // pub c9: u64,
}

#[derive(FromPrimitive)]
#[repr(u64)]
pub enum Method {
    Invoke = frc42_dispatch::method_hash!("Invoke"),
}
