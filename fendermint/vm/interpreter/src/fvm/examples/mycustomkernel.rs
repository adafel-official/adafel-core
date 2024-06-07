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
use fvm::syscalls::Memory;
use fvm::DefaultKernel;
use fvm_ipld_encoding::RawBytes;
use fvm_shared::clock::ChainEpoch;
use fvm_shared::randomness::RANDOMNESS_LENGTH;
use fvm_shared::sys::out::network::NetworkContext;
use fvm_shared::sys::out::vm::MessageContext;
use fvm_shared::{address::Address, econ::TokenAmount, ActorID, MethodNum};
use std::cmp;
use std::io::Read;

use ambassador::Delegate;
use cid::Cid;

pub trait CustomKernel: Kernel {
    fn my_custom_syscall(&self, data: &[u8], conv: &[u8]) -> Result<Vec<Vec<i128>>>;
}

// our custom kernel extends the filecoin kernel
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
#[delegate(SendOps<K>, generics = "K", where = "K: CustomKernel")]
#[delegate(UpgradeOps<K>, generics = "K", where = "K: CustomKernel")]
pub struct CustomKernelImpl<C>(pub DefaultKernel<C>);

impl<C> CustomKernel for CustomKernelImpl<C>
where
    C: CallManager,
    CustomKernelImpl<C>: Kernel,
{
    fn my_custom_syscall(&self, data: &[u8], conv: &[u8]) -> Result<Vec<Vec<i128>>> {
        // Here we have access to the Kernel structure and can call
        // any of its methods, send messages, etc.

        // We can also run an external program, link to any rust library
        // access the network, etc.
        let deserialized_data: Vec<Vec<i128>> = match fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(data)),
        ) {
            Err(e) => {
                vec![vec![0]]
            }
            Ok(r) => r,
        };

        let deserialized_conv: Vec<Vec<i128>> = match fvm_ipld_encoding::RawBytes::deserialize(
            &fvm_ipld_encoding::RawBytes::new(Vec::from(conv)),
        ) {
            Err(e) => {
                vec![vec![0]]
            }
            Ok(r) => r,
        };

        let mut output: Vec<Vec<i128>> =
            vec![
                vec![0; deserialized_data[0].len() - deserialized_conv[0].len() + 1];
                deserialized_data.len() - deserialized_conv.len() + 1
            ];

        for i in 0..(deserialized_data.len() - deserialized_conv.len() + 1) {
            for j in 0..(deserialized_data[0].len() - deserialized_conv[0].len() + 1) {
                let mut sum = 0;
                for ci in 0..(deserialized_conv.len()) {
                    for cj in 0..(deserialized_conv[0].len()) {
                        sum += deserialized_conv[ci][cj] * deserialized_data[i + ci][j + cj];
                    }
                }

                output[i][j] = sum;
            }
        }

        Ok(output)
    }
}

impl<C> Kernel for CustomKernelImpl<C>
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
        CustomKernelImpl(DefaultKernel::new(
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

impl<K> SyscallHandler<K> for CustomKernelImpl<K::CallManager>
where
    K: CustomKernel
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

        linker.link_syscall("my_custom_kernel", "my_custom_syscall", my_custom_syscall)?;

        Ok(())
    }
}

pub fn my_custom_syscall(
    context: fvm::syscalls::Context<'_, impl CustomKernel>,
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
    let result = context.kernel.my_custom_syscall(array, conv_array)?;

    let ser_result_raw = match fvm_ipld_encoding::RawBytes::serialize(result) {
        Err(e) => RawBytes::new(vec![0]),
        Ok(r) => r,
    };

    let ser_result: &[u8] = ser_result_raw.bytes();

    let output = context.memory.try_slice_mut(output_offset, output_length)?;
    let length = cmp::min(output.len(), ser_result.len());
    output[..length].copy_from_slice(ser_result);

    Ok(length as u32)
}
