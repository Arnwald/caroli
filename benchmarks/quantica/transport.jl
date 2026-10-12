# Quantica.jl runner for the shared transport problems of `benchmarks/transport.toml`.
#
# The system is rebuilt from the files written by `caroli-bench prepare` (the same Hamiltonian
# as every other code): the clean model on the infinite lattice, cut to the sample, with the
# on-site disorder added by a modifier; the leads are the clean model cut to one column and
# solved by `GS.Schur`. The device is solved by `GS.SparseLU` (UMFPACK), Quantica's solver for
# finite systems. Quantica adds `im·sqrt(eps)` to real energies.
#
# The first call of each function compiles it: a small instance of the problem runs first, and
# only the second build and its energies are timed. Julia runs on one thread, with one BLAS
# thread.
#
#     julia --project=quantica transport.jl DATA_DIR PROBLEM WIDTH

using Quantica, JSON, LinearAlgebra

BLAS.set_num_threads(1)

function readmatrix(m)
    # JSON: rows of [re, im] pairs.
    n = length(m)
    M = zeros(ComplexF64, n, length(m[1]))
    for i in 1:n, j in 1:length(m[1])
        M[i, j] = complex(m[i][j][1], m[i][j][2])
    end
    return M
end

function system(spec, data, name, width)
    n = spec["orbitals"]
    len = spec["length"] === nothing ? width : spec["length"]
    subs = spec["sublattices"]
    nsub = Base.length(subs)
    a1 = SVector{2}(Float64.(spec["vectors"][1]))
    a2 = SVector{2}(Float64.(spec["vectors"][2]))
    A = hcat(a1, a2)
    offs = [SVector{2}(Float64.(s["offset"])) for s in subs]
    names = Tuple(Symbol("s$k") for k in 1:nsub)
    lat = lattice([sublat(Tuple(offs[k]); name = names[k]) for k in 1:nsub]...; bravais = (Tuple(a1), Tuple(a2)))
    sm(M) = n == 1 ? M[1, 1] : SMatrix{n,n}(M)
    model = sum(onsite(sm(readmatrix(subs[k]["onsite"])); sublats = names[k]) for k in 1:nsub)
    for hop in spec["hoppings"]
        f, t = hop["from"] + 1, hop["to"] + 1
        d = Tuple(hop["offset"])
        M = readmatrix(hop["matrix"])
        # <(R, f)| H |(R + d, t)> = M: the hop from (R + d, t) to (R, f), and its reverse.
        model += hopping(sm(M); sublats = names[t] => names[f], dcells = .-d)
        model += hopping(sm(Matrix(M')); sublats = names[f] => names[t], dcells = d)
    end
    h = hamiltonian(lat, model; orbitals = n)
    # Cell coordinates (x, y) of a site at r on sublattice s.
    cell(r, s) = round.(Int, A \ (SVector{2}(r[1], r[2]) - offs[s]))
    insample(r) = any(s -> (c = cell(r, s); norm(A * c + offs[s] - r) < 1e-6 && 0 <= c[1] < len && 0 <= c[2] < width), 1:nsub)
    device = supercell(h; region = insample)
    if spec["anderson"] !== nothing
        raw = reinterpret(Float64, read("$data/$name/disorder_W$width.f64"))
        pot = reshape(raw, nsub, width, len) # Julia is column-major: index [s, y, x]
        function shift(r)
            for s in 1:nsub
                c = cell(r, s)
                if norm(A * c + offs[s] - r) < 1e-6
                    return pot[s, c[2] + 1, c[1] + 1]
                end
            end
            error("site not in the sample")
        end
        device = hamiltonian(device, @onsite!((o, r) -> o + shift(r) * I))()
    end
    column(r) = any(s -> (c = cell(r, s); norm(A * c + offs[s] - r) < 1e-6 && 0 <= c[2] < width && c[1] == 0), 1:nsub)
    lead = supercell(h, (1, 0); region = column) |> greenfunction(GS.Schur(boundary = 0))
    # Quantica attaches a lead on the sites of the end column that couple to the next column
    # (the lead surface): `(y, s)` pairs, for the left (x = 0) and right (x = len - 1) ends.
    surface_left, surface_right = Set{Tuple{Int,Int}}(), Set{Tuple{Int,Int}}()
    for hop in spec["hoppings"], y in 0:width-1
        f, t = hop["from"] + 1, hop["to"] + 1
        dx, dy = hop["offset"]
        # Site (x, y, f) couples to (x + dx, y + dy, t).
        0 <= y + dy < width || continue
        if dx == 1
            push!(surface_right, (y, f))
            push!(surface_left, (y + dy, t))
        elseif dx == -1
            push!(surface_left, (y, f))
            push!(surface_right, (y + dy, t))
        end
    end
    function at_end(r, x, surface)
        any(1:nsub) do s
            c = cell(r, s)
            norm(A * c + offs[s] - r) < 1e-6 && c[1] == x && (c[2], s) in surface
        end
    end
    first(r) = at_end(r, 0, surface_left)
    last(r) = at_end(r, len - 1, surface_right)
    g = device |> attach(lead; region = first, reverse = true) |> attach(lead; region = last) |> greenfunction(GS.SparseLU())
    return transmission(g[2, 1]), len
end

function main()
    data, name, width = ARGS[1], ARGS[2], parse(Int, ARGS[3])
    spec = JSON.parsefile("$data/$name/problem.json")
    spec["geometry"] === nothing || error("only two-terminal problems")
    # Compile everything on the smallest width of every problem (32) first.
    T, _ = system(spec, data, name, 32)
    T(spec["energies"][1])
    t0 = time()
    T, len = system(spec, data, name, width)
    build = time() - t0
    times, ts = Float64[], Float64[]
    for e in spec["energies"]
        t = time()
        push!(ts, T(e))
        push!(times, time() - t)
    end
    println(JSON.json(Dict(
        "code" => "quantica", "version" => string(pkgversion(Quantica)), "solver" => "GS.SparseLU, GS.Schur leads",
        "problem" => name, "width" => width, "length" => len, "orbitals" => spec["orbitals"], "threads" => 1,
        "build_s" => build, "times_s" => times, "energies" => spec["energies"], "transmissions" => ts,
    )))
end

main()
