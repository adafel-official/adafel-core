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
use fvm_shared::clock::ChainEpoch;
use fvm_shared::randomness::RANDOMNESS_LENGTH;
use fvm_shared::sys::out::network::NetworkContext;
use fvm_shared::sys::out::vm::MessageContext;
use fvm_shared::{address::Address, econ::TokenAmount, ActorID, MethodNum};

use ambassador::Delegate;
use cid::Cid;

// we define a single custom syscall which simply doubles the input
pub trait CustomKernel: Kernel {
    fn my_custom_syscall(
        &self,
        i1: u64,
        i2: u64,
        i3: u64,
        // i4: u64,
        // i5: u64,
        // i6: u64,
        // i7: u64,
        // i8: u64,
        // i9: u64,
        c1: u64,
        c2: u64,
        c3: u64,
        // c4: u64,
        // c5: u64,
        // c6: u64,
        // c7: u64,
        // c8: u64,
        // c9: u64,
    ) -> Result<u64>;
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
    fn my_custom_syscall(
        &self,
        i1: u64,
        i2: u64,
        i3: u64,
        // i4: u64,
        // i5: u64,
        // i6: u64,
        // i7: u64,
        // i8: u64,
        // i9: u64,
        c1: u64,
        c2: u64,
        c3: u64,
        // c4: u64,
        // c5: u64,
        // c6: u64,
        // c7: u64,
        // c8: u64,
        // c9: u64,
    ) -> Result<u64> {
        // Here we have access to the Kernel structure and can call
        // any of its methods, send messages, etc.

        // We can also run an external program, link to any rust library
        // access the network, etc.
        let result = i1 * c1 + i2 * c2 + i3 * c3; //+ i4 * c4 + i5 * c5 + i6 * c6 + i7 * c7 + i8 * c8 + i9 * c9;
        Ok(result)
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
    i1: u64,
    i2: u64,
    i3: u64,
    // i4: u64,
    // i5: u64,
    // i6: u64,
    // i7: u64,
    // i8: u64,
    // i9: u64,
    c1: u64,
    c2: u64,
    c3: u64,
    // c4: u64,
    // c5: u64,
    // c6: u64,
    // c7: u64,
    // c8: u64,
    // c9: u64,
) -> Result<u64> {
    context.kernel.my_custom_syscall(i1, i2, i3, c1, c2, c3)
}
