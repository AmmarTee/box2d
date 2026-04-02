# Box2D v3 — Project Guidelines

## Code Style

- **Language**: C17 with extensions (anonymous unions, static assert). Samples use C++20.
- **Naming**: `b2` prefix everywhere — `b2CreateWorld()` (functions), `b2Vec2` (types), `B2_NULL_INDEX` (macros), `b2_nullBodyId` (constants), `b2BodyDef` (definitions)
- **Headers**: SPDX license header (`// SPDX-FileCopyrightText: 2023 Erin Catto` + `// SPDX-License-Identifier: MIT`), include guards
- **Memory**: Use `B2_ALLOC_STRUCT()`, `B2_ALLOC_ARRAY()`, `B2_FREE_STRUCT()`, `B2_FREE_ARRAY()` — never raw malloc/free in library code
- **Assertions**: `B2_ASSERT()` macro for invariant checks
- **Visibility**: All symbols hidden by default; export with `B2_API` macro
- **Floating point**: `-ffp-contract=off` enforced for determinism — do not use FMA intrinsics unless guarded

## Architecture

- **Data-oriented design**: Contiguous arrays for bodies/shapes/joints/contacts, free-list reuse, no per-frame allocations after init
- **Public API** in `include/box2d/` (7 headers) — users include only `box2d/box2d.h`
- **Internal implementation** in `src/` — not exposed to consumers
- **Multithreading**: Data-parallel task system via user-provided callbacks (`b2EnqueueTaskCallback`). World is NOT thread-safe during `b2World_Step()`
- **SIMD**: Auto-detected (AVX2/SSE2/NEON/None) via `src/core.h` platform macros. Width varies (4 or 8).
- See [docs/](docs/) for detailed documentation on simulation, collision, joints, and migration

## Build and Test

```bash
# Windows — generate Visual Studio solution
create_sln.bat
# Then build in VS or: cmake --build build --config Release

# Linux/macOS
mkdir build && cd build && cmake .. && cmake --build .

# Run tests (after building)
./build/bin/test          # Linux/macOS
build\bin\test.exe        # Windows
```

### Key CMake options
| Option | Default | Purpose |
|--------|---------|---------|
| `BOX2D_DISABLE_SIMD` | OFF | Disable SIMD math |
| `BOX2D_AVX2` | OFF | Enable AVX2 (x86_64) |
| `BOX2D_UNIT_TESTS` | ON | Build unit tests |
| `BOX2D_SAMPLES` | ON | Build interactive samples |
| `BOX2D_BENCHMARKS` | OFF | Build benchmarks |
| `BOX2D_VALIDATE` | ON | Heavy validation |

## Conventions

- **Sentinel value**: `B2_NULL_INDEX` (-1) for invalid indices — never use 0 or nullptr for index slots
- **ID types** are opaque structs (`b2BodyId`, `b2ShapeId`, etc.) — compare with dedicated functions, not raw equality
- **Tests**: Simple `XXXTest()` functions in `test/test_*.c`, registered via `RUN_TEST()` macro in `test/main.c`. No external test framework.
- **Shared code** (`shared/`) is linked by both tests and samples — put reusable non-library code there
- **No C++ in library source** — `src/` is pure C17. C++ only in `samples/`
