# ez-gfx-assets

Compiler-free texture asset validation and staging primitives. It parses direct KTX2 BC/ASTC levels, hands supported BasisLZ payloads to the optional `basis` transcoder, tracks mip residency, and provides unbounded CPU admission and event delivery subject to real allocation/OS failure. It emits validated payloads and host-polled outcomes without claiming GPU upload success.
