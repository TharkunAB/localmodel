# Native candidate source review

This is a bounded source review. It does not establish memory safety, a successful
build, numerical parity, or GPU compatibility.

## Scope and provenance

The base is PrismML `e3de715222b2667c54a267441159a5729f11155e`.
The source archive is pinned by its SHA-256 in `manifest.json`.
The selected upstream patches come from bonsai-ada-surgery
`f0a11b8b7ac5e82dcfdcd48015beb9eabf9501d2`.
The original patch files retain their author and license context.

The combined patch changes 17 existing native C++ or CUDA files.
It changes no CMake files, Python tools, proxy routes, model weights, remote endpoints,
client permissions, tool execution paths, or reasoning effort rules.
Inspection found no new network calls, telemetry, subprocess calls, downloads, or URL parsing.
The existing server still has its original network features. Wrapper configuration controls those features.

## Draft window and target verification

The MTP context receives a smaller `n_ctx` only when MTP and a nonzero draft window are set.
The size includes padding for prompt batches and verification work.
The size calculation uses 64-bit arithmetic.
The target context size is read but is not changed.

Before the draft receives a target batch, the server removes old rows from `ctx_dft`.
It does not remove target rows. It keeps absolute positions.
The removal uses the lowest position in the batch, so it retains enough rows for that batch.
The check requires MTP, a nonempty batch, and a positive window.
The existing target sampler and verification path remain in use.

The original 0031 tail cap was omitted. It would cap lookup drafts at the MTP length.
The original stopping-rule hunk was also omitted. Its repeated context matched Eagle3
when applied to this newer Prism source. Eagle3 is unchanged in the final combined patch.
The existing MTP chaining code is unchanged.

This review does not prove that a truncated draft cache preserves acceptance or every
state restore path. Check long prompts, cancellation, cache restore, compaction, and recall later.
The target verification step limits the effect of a weaker draft. Kernel arithmetic
and speculative sampling can still change generated text.

## Tier allocation and lifetime

The tiered allocator uses CUDA virtual-memory mappings. Each target layer receives a
VRAM prefix and a host-backed suffix in the same virtual address range.
The selected code calls CUDA allocation and copy APIs. It does not write host data to a file or network.
The host allocation needs driver support for `CU_MEM_LOCATION_TYPE_HOST`.
That support has not been checked on this WSL2 driver.

The release callback synchronizes the device before it removes aliases and releases
the virtual mappings and allocation handles.
Patch 0041 gives the target and draft caches separate staging keys.
The pipeline uses CUDA events to order reads, copies, and write-back operations.
It restores temporary tensor data pointers after the relevant work.
Backend-context destruction removes the pipeline events and state entries.

Staging allocations remain in a process-wide cache for the life of the process.
The global tier-entry registry returns pointers after releasing its mutex.
This is a review concern for concurrent allocation, model hot reload, or buffer destruction.
The prepared use case is one GPU, one model lifetime, and one slot.
Do not claim general concurrency safety from this review.

The tier buffer applies only when the backend, layout, K/V types, and cache structure qualify.
An unsupported path can retain an ordinary device buffer. Check the actual allocation log.
A successful parameter parse alone does not prove that the host tail is active.

## Graph adaptation

Most of 0041 applies unchanged. Two graph hooks depended on optional upstream operation-timing code.
The candidate places the same begin and step calls after the existing gather resets
and before the node loop. It places the end call after that loop.
The pipeline stays disabled for concurrent-event graphs and CUDA graph capture.
Patch 0053 selects the non-captured pipeline when attention reaches the host tail.
The ordinary CUDA graph path remains available below the boundary.

The static review checked the hook placement and pointer restoration.
It did not execute the scheduling path or test CUDA event ordering.

## Build and runtime gates

- Configure and compile with the manifest's private CUDA 13 toolkit and Ada architecture.
- Check native help for the expected flags and the binary's runtime libraries.
- Check source hashes and build identity before launch.
- Check actual target and draft cache sizes and VRAM/host allocations.
- Run small tool and grammar checks before the long conversation comparison.
- Check full-context recall, compaction, saved-state restore, and cancellation.
- Compare output and acceptance with the stock runtime for each isolated change.
- Measure performance only after the machine is available.

None of these native build or runtime gates has run during preparation.
