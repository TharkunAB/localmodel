"""Block outbound sockets before the Strata server or its children start."""

from __future__ import annotations

import ctypes
import errno
import os
import socket
import sys


class Comparison(ctypes.Structure):
    """Use the argument comparison layout from libseccomp."""

    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


def restrict_network() -> None:
    """Retain inbound API replies. Deny new connections and addressed sends."""
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                          ctypes.c_uint, ctypes.POINTER(Comparison)]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("libseccomp could not create the network policy.")

    def deny(name: str, *comparisons: Comparison) -> None:
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if number < 0:
            raise RuntimeError(f"libseccomp does not know the required syscall: {name}.")
        args = (Comparison * len(comparisons))(*comparisons)
        result = lib.seccomp_rule_add_array(context, 0x00050000 | errno.EPERM,
                                            number, len(comparisons), args)
        if result != 0:
            raise RuntimeError(f"Cannot restrict {name}: {os.strerror(-result)}.")

    try:
        # Children retain the filter across fork and exec. Block alternate I/O paths.
        for name in ("connect", "sendmsg", "sendmmsg", "recvmsg", "recvmmsg", "io_uring_setup", "io_uring_enter",
                     "io_uring_register", "ptrace", "process_vm_writev", "pidfd_getfd"):
            deny(name)
        # send() uses sendto() with a null destination on accepted API sockets.
        deny("sendto", Comparison(4, 1, 0, 0))  # SCMP_CMP_NE
        # Exclude vsock, packet sockets, netlink and raw sockets.
        for family in range(socket.AF_INET6):
            if family not in (socket.AF_UNIX, socket.AF_INET):
                deny("socket", Comparison(0, 4, family, 0))  # SCMP_CMP_EQ
        deny("socket", Comparison(0, 6, socket.AF_INET6, 0))  # SCMP_CMP_GT
        for kind in range(16):
            if kind == socket.SOCK_SEQPACKET:
                # Linux CUDA needs a local sequenced-packet listener for UVM.
                deny("socket", Comparison(1, 7, 0xF, kind), Comparison(0, 1, socket.AF_UNIX, 0))
            elif kind != socket.SOCK_STREAM:
                deny("socket", Comparison(1, 7, 0xF, kind))  # SCMP_CMP_MASKED_EQ
        deny("socketpair", Comparison(0, 1, socket.AF_UNIX, 0))
        result = lib.seccomp_load(context)
        if result != 0:
            raise RuntimeError(f"Cannot load the network policy: {os.strerror(-result)}.")
    finally:
        lib.seccomp_release(context)


def main() -> None:
    """Close inherited descriptors, install the policy, and execute the command."""
    if len(sys.argv) < 2:
        raise RuntimeError("A command is required.")
    descriptors = [int(name) for name in os.listdir("/proc/self/fd") if name.isdecimal() and int(name) > 2]
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass
    restrict_network()
    print("[localmodel] Strata: outgoing connections blocked for the server and its children.",
          file=sys.stderr, flush=True)
    os.execv(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        sys.exit(f"Strata isolation failed. The server was not started: {error}")
