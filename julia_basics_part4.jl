# Julia Basics Part 4: Project Organization, Modules, and Custom Types
# This program covers project structure, module creation, custom types,
# and demonstrates building a reusable statistics module
# Topics: Project organization, modules, packages, custom types, 
#         descriptive statistics, statistical visualization

println("=== Julia Basics Part 4: Project Organization, Modules, and Custom Types ===")
println()

# ============================================================================
# SECTION 1: PROJECT ORGANIZATION IN JULIA
# ============================================================================
println("# ================ 1. PROJECT ORGANIZATION ================")

# Julia projects follow a standard structure:
#
# MyProject/
# ├── Project.toml          # Project metadata and dependencies
# ├── Manifest.toml         # Exact versions of all dependencies
# ├── src/                  # Source code
# │   ├── MyModule.jl       # Module files
# │   └── ...
# ├── test/                 # Test files
# │   ├── runtests.jl
# │   └── ...
# ├── docs/                 # Documentation
# │   ├── make.jl
# │   └── src/
# └── README.md             # Project description

println("""
Julia Project Structure:

MyProject/
├── Project.toml          # Project metadata and dependencies
├── Manifest.toml         # Exact versions of all dependencies
├── src/                  # Source code
│   ├── MyModule.jl       # Module files
│   └── ...
├── test/                 # Test files
│   ├── runtests.jl
│   └── ...
├── docs/                 # Documentation
│   ├── make.jl
│   └── src/
└── README.md             # Project description
""")

# Creating a project using Pkg
println("# Creating a project:")
println("  In REPL: ] generate MyProject")
println("  Or: using Pkg; Pkg.generate(\"MyProject\")")

# Project.toml example content
println("\n# Example Project.toml:")
println("""
[deps]
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
""")

# ============================================================================
# SECTION 2: MODULES AND PACKAGES
# ============================================================================
println("\n# ================ 2. MODULES AND PACKAGES ================")

# ----------------------------------------------------------------------------
# 2.1 UNDERSTANDING MODULES
# ----------------------------------------------------------------------------
println("\n# --- 2.1 UNDERSTANDING MODULES ---")
#
# Modules in Julia are namespaces that organize code
# They help avoid name conflicts and organize related functionality

# Basic module structure
module ExampleModule
    # Exported functions (visible outside module)
    export greet, calculate_sum
    
    # Internal function (not exported)
    function _internal_helper(x)
        return x * 2
    end
    
    # Exported functions
    greet(name) = "Hello, $name !"
    calculate_sum(a, b) = a + b
end

println("Module created: ExampleModule")
println("Greet from module: ", ExampleModule.greet("User"))
println("Calculate sum: ", ExampleModule.calculate_sum(5, 3))

# ----------------------------------------------------------------------------
# 2.2 USING AND IMPORTING
# ----------------------------------------------------------------------------
println("\n# --- 2.2 USING AND IMPORTING ---")

# Different ways to use modules:
#
# 1. using ModuleName - brings all exported names into scope
# 2. using ModuleName: name1, name2 - brings specific names
# 3. import ModuleName - brings module into scope (use ModuleName.name)
# 4. import ModuleName: name1, name2 - import specific names for extension

# Example with Statistics module
using Statistics

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
println("Data: ", data)
println("Mean: ", mean(data))
println("Median: ", median(data))
println("Std: ", std(data))
println("Var: ", var(data))

# ----------------------------------------------------------------------------
# 2.3 CREATING CUSTOM MODULES
# ----------------------------------------------------------------------------
println("\n# --- 2.3 CREATING CUSTOM MODULES ---")

# We'll create a comprehensive Statistics module below
# This demonstrates best practices for module creation

# ============================================================================
# SECTION 3: CREATING A STATISTICS MODULE
# ============================================================================
println("\n# ================ 3. CREATING A STATISTICS MODULE ================")

# Define our custom StatisticsTools module
module StatisticsTools

# Export public API
export 
    # Types
    StatisticalSummary,
    # Functions
    describe,
    summary_stats,
    outlier_detection,
    normalize,
    standardize,
    # Plotting functions
    plot_histogram,
    plot_boxplot,
    plot_qq,
    plot_distribution,
    # Utility functions
    missing_count,
    unique_count,
    correlation_matrix

# Import required packages
using Statistics
using StatsBase
using Plots
using Distributions
using LinearAlgebra

# ============================================================================
# CUSTOM TYPES
# ============================================================================

"""
    StatisticalSummary

A struct containing comprehensive descriptive statistics for a dataset.

# Fields
- `n::Int`: Number of observations
- `missing_count::Int`: Number of missing values
- `mean::Float64`: Arithmetic mean
- `median::Float64`: Median value
- `std::Float64`: Standard deviation
- `var::Float64`: Variance
- `min::Float64`: Minimum value
- `max::Float64`: Maximum value
- `q25::Float64`: 25th percentile
- `q75::Float64`: 75th percentile
- `skewness::Float64`: Skewness coefficient
- `kurtosis::Float64`: Kurtosis coefficient
- `cv::Float64`: Coefficient of variation (std/mean)
"""
struct StatisticalSummary
    n::Int
    missing_count::Int
    mean::Float64
    median::Float64
    std::Float64
    var::Float64
    min::Float64
    max::Float64
    q25::Float64
    q75::Float64
    skewness::Float64
    kurtosis::Float64
    cv::Float64
end

# Custom show method for pretty printing
function Base.show(io::IO, s::StatisticalSummary)
    println(io, "StatisticalSummary:")
    println(io, "  Observations:     $(s.n)")
    println(io, "  Missing:          $(s.missing_count)")
    println(io, "  Mean:             $(round(s.mean, digits=4))")
    println(io, "  Median:           $(round(s.median, digits=4))")
    println(io, "  Std Dev:          $(round(s.std, digits=4))")
    println(io, "  Variance:         $(round(s.var, digits=4))")
    println(io, "  Min:              $(round(s.min, digits=4))")
    println(io, "  Max:              $(round(s.max, digits=4))")
    println(io, "  Q25 (25%):        $(round(s.q25, digits=4))")
    println(io, "  Q75 (75%):        $(round(s.q75, digits=4))")
    println(io, "  Skewness:         $(round(s.skewness, digits=4))")
    println(io, "  Kurtosis:         $(round(s.kurtosis, digits=4))")
    println(io, "  Coeff. Variation: $(round(s.cv, digits=4))")
end

# ============================================================================
# DESCRIPTIVE STATISTICS FUNCTIONS
# ============================================================================

"""
    summary_stats(data::Vector{<:Real})

Calculate comprehensive descriptive statistics for a vector.

# Arguments
- `data`: Input vector of real numbers

# Returns
- `StatisticalSummary`: Struct containing all statistics
"""
function summary_stats(data::Vector{<:Real})
    # Handle missing values
    clean_data = filter(!isnan, data)
    n_missing = length(data) - length(clean_data)
    n = length(data)
    
    if length(clean_data) == 0
        error("No valid data points")
    end
    
    # Basic statistics
    mean_val = mean(clean_data)
    median_val = median(clean_data)
    std_val = std(clean_data)
    var_val = var(clean_data)
    min_val = minimum(clean_data)
    max_val = maximum(clean_data)
    
    # Quantiles
    q25 = quantile(clean_data, 0.25)
    q75 = quantile(clean_data, 0.75)
    
    # Skewness and Kurtosis (using StatsBase)
    skew_val = skewness(clean_data)
    kurt_val = kurtosis(clean_data)
    
    # Coefficient of variation
    cv_val = mean_val != 0 ? std_val / abs(mean_val) : NaN
    
    return StatisticalSummary(
        n, n_missing, mean_val, median_val, std_val, var_val,
        min_val, max_val, q25, q75, skew_val, kurt_val, cv_val
    )
end

"""
    describe(data::Vector{<:Real})

Print a formatted description of the data statistics.

# Arguments
- `data`: Input vector
"""
function describe(data::Vector{<:Real})
    stats = summary_stats(data)
    show(stdout, stats)
    return stats
end

"""
    outlier_detection(data::Vector{<:Real}; method::String="iqr")

Detect outliers in the data.

# Arguments
- `data`: Input vector
- `method`: "iqr" (Interquartile Range) or "zscore" (Z-score > 3)

# Returns
- `Tuple`: (outlier_indices, outlier_values, clean_data)
"""
function outlier_detection(data::Vector{<:Real}; method::String="iqr")
    clean_data = filter(!isnan, data)
    
    if method == "iqr"
        q1 = quantile(clean_data, 0.25)
        q3 = quantile(clean_data, 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = (clean_data .< lower_bound) .| (clean_data .> upper_bound)
    elseif method == "zscore"
        mean_val = mean(clean_data)
        std_val = std(clean_data)
        z_scores = abs.((clean_data .- mean_val) ./ std_val)
        outlier_mask = z_scores .> 3
    else
        error("Unknown method: $method. Use 'iqr' or 'zscore'")
    end
    
    outlier_indices = findall(outlier_mask)
    outlier_values = clean_data[outlier_mask]
    clean_values = clean_data[.!outlier_mask]
    
    return outlier_indices, outlier_values, clean_values
end

"""
    normalize(data::Vector{<:Real})

Normalize data to [0, 1] range (min-max scaling).

# Arguments
- `data`: Input vector

# Returns
- `Vector{Float64}`: Normalized data
"""
function normalize(data::Vector{<:Real})
    clean_data = float.(filter(!isnan, data))
    min_val = minimum(clean_data)
    max_val = maximum(clean_data)
    
    if max_val == min_val
        return zeros(length(clean_data))
    end
    
    return (clean_data .- min_val) ./ (max_val .- min_val)
end

"""
    standardize(data::Vector{<:Real})

Standardize data (z-score normalization).

# Arguments
- `data`: Input vector

# Returns
- `Vector{Float64}`: Standardized data (mean=0, std=1)
"""
function standardize(data::Vector{<:Real})
    clean_data = float.(filter(!isnan, data))
    mean_val = mean(clean_data)
    std_val = std(clean_data)
    
    if std_val == 0
        return zeros(length(clean_data))
    end
    
    return (clean_data .- mean_val) ./ std_val
end

"""
    missing_count(data::Vector)

Count missing values (NaN, nothing, missing).

# Arguments
- `data`: Input vector

# Returns
- `Int`: Count of missing values
"""
function missing_count(data::Vector)
    count = 0
    for x in data
        if ismissing(x) || isnothing(x) || (isa(x, Number) && isnan(x))
            count += 1
        end
    end
    return count
end

"""
    unique_count(data::Vector)

Count unique values in the data.

# Arguments
- `data`: Input vector

# Returns
- `Int`: Number of unique values
"""
function unique_count(data::Vector)
    return length(unique(data))
end

"""
    correlation_matrix(data::Matrix{<:Real})

Calculate correlation matrix for multiple variables.

# Arguments
- `data`: Matrix where columns are variables

# Returns
- `Matrix{Float64}`: Correlation matrix
"""
function correlation_matrix(data::Matrix{<:Real})
    return cor(data)
end

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

"""
    plot_histogram(data::Vector{<:Real}; kwargs...)

Create a histogram with density curve.

# Arguments
- `data`: Input vector
- `kwargs`: Additional plotting arguments
"""
function plot_histogram(data::Vector{<:Real}; kwargs...)
    clean_data = filter(!isnan, data)
    
    # Create histogram
    p = histogram(clean_data, 
                  normalize=true,
                  alpha=0.6,
                  label="Histogram",
                  xlabel="Value",
                  ylabel="Density",
                  title="Histogram with Density Curve",
                  color=:blue)
    
    # Add density curve
    density_curve = Normal(mean(clean_data), std(clean_data))
    x_range = range(minimum(clean_data), maximum(clean_data), length=200)
    plot!(p, x_range, pdf.(density_curve, x_range), 
          linewidth=3, 
          label="Normal Fit",
          color=:red)
    
    return p
end

"""
    plot_boxplot(data::Vector{<:Real}; label::String="Data")

Create a box plot.

# Arguments
- `data`: Input vector
- `label`: Label for the plot
"""
function plot_boxplot(data::Vector{<:Real}; label::String="Данные")
    clean_data = filter(!isnan, data)
    
    fig = boxplot([label], [clean_data],
                  label="",
                  ylabel="Значение",
                  title="Box Plot",
                  color=:green,
                  size=(600, 400))
    
    return fig
end

"""
    plot_qq(data::Vector{<:Real})

Create a Q-Q plot (Quantile-Quantile) for normality check.

# Arguments
- `data`: Input vector
"""
function plot_qq(data::Vector{<:Real})
    clean_data = filter(!isnan, data)
    
    n = length(clean_data)
    theoretical = quantile.(Normal(), (1:n) ./ (n + 1))
    sample = sort(clean_data)
    theoretical = theoretical .* std(sample) .+ mean(sample)
    
    fig = scatter(theoretical, sample,
                  label="Данные",
                  xlabel="Теоретические квантили",
                  ylabel="Выборочные квантили",
                  title="Q-Q Plot (проверка нормальности)",
                  color=:purple,
                  size=(600, 600))
    
    min_val = min(minimum(theoretical), minimum(sample))
    max_val = max(maximum(theoretical), maximum(sample))
    plot!(fig, [min_val, max_val], [min_val, max_val],
          linewidth=2,
          linestyle=:dash,
          label="Опорная линия",
          color=:red)
    
    return fig
end

"""
    plot_distribution(data::Vector{<:Real})

Create a comprehensive distribution plot (histogram + boxplot + density).

# Arguments
- `data`: Input vector
"""
function plot_distribution(data::Vector{<:Real})
    clean_data = filter(!isnan, data)
    
    # Histogram
    fig_hist = histogram(clean_data, 
                         normalize=true,
                         alpha=0.6,
                         label="Histogram",
                         xlabel="Value",
                         ylabel="Density",
                         title="Histogram",
                         color=:blue,
                         legend=:topright)
    
    density_curve = Normal(mean(clean_data), std(clean_data))
    x_range = range(minimum(clean_data), maximum(clean_data), length=200)
    plot!(fig_hist, x_range, pdf.(density_curve, x_range), 
          linewidth=2, 
          label="Normal Fit",
          color=:red)
    
    # Box plot
    fig_box = boxplot(["Data"], [clean_data],
                      label="",
                      ylabel="Value",
                      title="Box Plot",
                      color=:green)
    
    # Sorted values
    fig_sorted = plot(sort(clean_data),
                      label="Sorted Data",
                      xlabel="Index",
                      ylabel="Value",
                      title="Sorted Values",
                      color=:purple,
                      linewidth=2)
    
    # Combine plots
    combined = plot(fig_hist, fig_box, fig_sorted, layout=(1, 3), size=(1200, 400))
    
    return combined
end

end # module StatisticsTools

println("Module StatisticsTools created successfully!")

# ============================================================================
# SECTION 4: USING THE STATISTICS MODULE
# ============================================================================
println("\n# ================ 4. USING THE STATISTICS MODULE ================")

# Import our custom module
using .StatisticsTools

# Generate sample data
using Random
Random.seed!(42)

# Create different distributions for testing
normal_data = randn(500) .* 10 .+ 50      # Normal: mean=50, std=10
skewed_data = randexp(500) .* 5           # Exponential (skewed)
uniform_data = rand(500) .* 100           # Uniform: 0-100

# Add some outliers and missing values
normal_data_with_outliers = vcat(normal_data, [150, 160, -50])
normal_data_with_missing = vcat(normal_data, [NaN, NaN, NaN])

println("\n--- Testing with Normal Distribution ---")
println("Data length: ", length(normal_data))

# Get descriptive statistics
stats_normal = StatisticsTools.describe(normal_data)

# Create plots
println("\nGenerating plots...")

# Histogram with density
plot_hist = StatisticsTools.plot_histogram(normal_data)
display(plot_hist)
savefig(plot_hist, "stats_histogram.png")
println("Saved: stats_histogram.png")

# Box plot
plot_box = StatisticsTools.plot_boxplot(normal_data, label="Normal Data")
display(plot_box)
savefig(plot_box, "stats_boxplot.png")
println("Saved: stats_boxplot.png")

# Q-Q plot
plot_qq_plot = StatisticsTools.plot_qq(normal_data)
display(plot_qq_plot)
savefig(plot_qq_plot, "stats_qq.png")
println("Saved: stats_qq.png")

# Comprehensive distribution plot
plot_dist = StatisticsTools.plot_distribution(normal_data)
display(plot_dist)
savefig(plot_dist, "stats_distribution.png")
println("Saved: stats_distribution.png")

# Test outlier detection
println("\n--- Outlier Detection ---")
outlier_indices, outlier_values, clean_values = StatisticsTools.outlier_detection(
    normal_data_with_outliers, 
    method="iqr"
)
println("Outliers found: ", length(outlier_values))
println("Outlier values: ", outlier_values)
println("Clean data length: ", length(clean_values))

# Test normalization
println("\n--- Normalization ---")
normalized = StatisticsTools.normalize(normal_data)
println("Original range: [$(minimum(normal_data)), $(maximum(normal_data))]")
println("Normalized range: [$(minimum(normalized)), $(maximum(normalized))]")
println("Normalized mean: ", round(mean(normalized), digits=4))

# Test standardization
println("\n--- Standardization ---")
standardized = StatisticsTools.standardize(normal_data)
println("Standardized mean: ", round(mean(standardized), digits=4))
println("Standardized std: ", round(std(standardized), digits=4))

# Test missing count
println("\n--- Missing Values ---")
println("Missing count: ", StatisticsTools.missing_count(normal_data_with_missing))

# Test unique count
println("\n--- Unique Values ---")
discrete_data = rand(1:10, 100)
println("Unique values in discrete data: ", StatisticsTools.unique_count(discrete_data))

# Test correlation matrix
println("\n--- Correlation Matrix ---")
correlation_data = randn(100, 3)
correlation_data[:, 2] = correlation_data[:, 1] .* 0.8 .+ randn(100)
correlation_data[:, 3] = correlation_data[:, 1] .* 0.5 .+ randn(100)
corr_matrix = StatisticsTools.correlation_matrix(correlation_data)
println("Correlation matrix (3 variables):")
println(round.(corr_matrix, digits=3))

# ============================================================================
# SECTION 5: COMPARING DISTRIBUTIONS
# ============================================================================
println("\n# ================ 5. COMPARING DISTRIBUTIONS ================")

# Compare different distributions
println("\n--- Comparing Normal, Skewed, and Uniform Distributions ---")

stats_skewed = StatisticsTools.summary_stats(skewed_data)
stats_uniform = StatisticsTools.summary_stats(uniform_data)

println("\nNormal Distribution:")
println("  Mean: $(round(stats_normal.mean, digits=2)), Std: $(round(stats_normal.std, digits=2))")
println("  Skewness: $(round(stats_normal.skewness, digits=3)), Kurtosis: $(round(stats_normal.kurtosis, digits=3))")

println("\nSkewed (Exponential) Distribution:")
println("  Mean: $(round(stats_skewed.mean, digits=2)), Std: $(round(stats_skewed.std, digits=2))")
println("  Skewness: $(round(stats_skewed.skewness, digits=3)), Kurtosis: $(round(stats_skewed.kurtosis, digits=3))")

println("\nUniform Distribution:")
println("  Mean: $(round(stats_uniform.mean, digits=2)), Std: $(round(stats_uniform.std, digits=2))")
println("  Skewness: $(round(stats_uniform.skewness, digits=3)), Kurtosis: $(round(stats_uniform.kurtosis, digits=3))")

# Create comparison plot
plot_compare = histogram(normal_data, normalize=true, alpha=0.4, label="Normal", color=:blue)
histogram!(plot_compare, skewed_data, normalize=true, alpha=0.4, label="Skewed", color=:red)
histogram!(plot_compare, uniform_data, normalize=true, alpha=0.4, label="Uniform", color=:green)
plot!(plot_compare, xlabel="Value", ylabel="Density", title="Distribution Comparison", size=(800, 500))
display(plot_compare)
savefig(plot_compare, "stats_distribution_compare.png")
println("\nSaved: stats_distribution_compare.png")

# ============================================================================
# SECTION 6: RECOMMENDED JULIA PACKAGES
# ============================================================================
println("\n# ================ 6. RECOMMENDED JULIA PACKAGES ================")

println("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    RECOMMENDED JULIA PACKAGES                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  DATA STORAGE & TABLES                                                   ║
║  ─────────────────────                                                   ║
║  • DataFrames.jl        - DataFrame manipulation (like pandas/R)         ║
║  • CSV.jl               - Read/write CSV files                           ║
║  • ExcelFiles.jl        - Read/write Excel files                         ║
║  • JSON.jl              - JSON parsing and writing                       ║
║  • Arrow.jl             - Apache Arrow format for fast data exchange     ║
║  • SQLite.jl            - SQLite database interface                      ║
║  • JDF.jl               - Julia Data Frame format (fast native format)   ║
║                                                                          ║
║  MATHEMATICAL MODELING & SIMULATION                                      ║
║  ─────────────────────────────────────                                   ║
║  • DifferentialEquations.jl - Comprehensive ODE/PDE/SDE solvers          ║
║  • ModelingToolkit.jl   - Symbolic modeling and automatic differentiation║
║  • Catalyst.jl          - Chemical reaction network modeling             ║
║  • AgentBasedModels.jl  - Agent-based modeling framework                 ║
║  • DifferentialEquations.jl - Stochastic and delay differential eq.      ║
║                                                                          ║
║  EQUATION SOLVING                                                        ║
║  ──────────────────                                                      ║
║  • NLsolve.jl           - Non-linear equation solving                    ║
║  • LinearSolve.jl       - Linear system solvers                          ║
║  • Roots.jl             - Root finding algorithms                        ║
║  • Optim.jl             - Optimization algorithms                        ║
║  • JuMP.jl              - Mathematical optimization programming          ║
║                                                                          ║
║  STATISTICS & VISUALIZATION                                              ║
║  ─────────────────────────────                                           ║
║  • Plots.jl             - Plotting metapackage (unified interface)       ║
║  • Makie.jl             - High-performance interactive visualization     ║
║  • StatsBase.jl         - Basic statistical functions                    ║
║  • Distributions.jl     - Probability distributions                      ║
║  • HypothesisTests.jl   - Statistical hypothesis testing                 ║
║  • GLM.jl               - Generalized linear models                      ║
║  • MixedModels.jl       - Mixed effects models                           ║
║  • Turing.jl            - Bayesian inference with MCMC                   ║
║  • RDatasets.jl         - Collection of datasets for examples            ║
║                                                                          ║
║  MACHINE LEARNING                                                        ║
║  ───────────────────                                                     ║
║  • MLJ.jl               - Machine Learning framework (like scikit-learn) ║
║  • Flux.jl              - Deep learning library                          ║
║  • Metalhead.jl         - Pre-trained deep learning models              ║
║  • Clustering.jl        - Clustering algorithms                          ║
║  • DimensionalReduction.jl - PCA, t-SNE, UMAP                           ║
║                                                                          ║
║  TIME SERIES                                                             ║
║  ─────────────────                                                       ║
║  • TimeSeries.jl        - Time series data structures                    ║
║  • Temporal.jl          - Time series analysis                           ║
║  • ARCHModels.jl        - ARCH/GARCH models for volatility              ║
║  • StateSpaceModels.jl  - State space models and Kalman filters         ║
║                                                                          ║
║  BIOLOGY & MEDICINE SPECIFIC                                             ║
║  ─────────────────────────────                                           ║
║  • BioJulia (BioSequences.jl) - Biological sequence analysis             ║
║  • Pharmacometrics.jl   - Pharmacokinetic/pharmacodynamic modeling       ║
║  • SBML.jl              - Systems Biology Markup Language support        ║
║  • CellML.jl            - CellML model support                           ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# Installation commands
println("\n# Installation Commands:")
println("  using Pkg")
println("  Pkg.add([\"DataFrames\", \"CSV\", \"Plots\", \"DifferentialEquations\"])")
println("  Pkg.add([\"StatsBase\", \"Distributions\", \"HypothesisTests\"])")
println("  Pkg.add([\"Makie\", \"MLJ\", \"Flux\"])")

# ============================================================================
# SECTION 7: MAKIE VS PLOTS COMPARISON
# ============================================================================
println("\n# ================ 7. MAKIE VS PLOTS ================")

println("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    PLOTS.JL VS MAKIE.JL COMPARISON                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PLOTS.JL                                                               │
│  ─────────                                                              │
│  ✓ Mature and stable                                                    │
│  ✓ Simple API, easy to learn                                            │
│  ✓ Multiple backends (GR, PyPlot, PlotlyJS, etc.)                       │
│  ✓ Good for standard scientific plots                                   │
│  ✓ Lower learning curve                                                 │
│  ✗ Slower for large datasets                                            │
│  ✗ Limited interactivity                                                │
│  ✗ Less customizable for complex visualizations                         │
│                                                                         │
│  MAKIE.JL                                                               │
│  ─────────                                                              │
│  ✓ High performance (GPU-accelerated)                                   │
│  ✓ Highly interactive (zoom, pan, hover)                                │
│  ✓ Beautiful default styling                                            │
│  ✓ Excellent for complex 3D visualizations                              │
│  ✓ Real-time animations                                                 │
│  ✓ Modern OpenGL-based rendering                                        │
│  ✗ Steeper learning curve                                               │
│  ✗ Newer package (less community resources)                             │
│  ✗ Requires more setup                                                  │
│                                                                         │
│  RECOMMENDATION:                                                        │
│  ──────────────                                                         │
│  • Use Plots.jl for: Quick exploration, standard plots, reports         │
│  • Use Makie.jl for: Interactive dashboards, complex 3D, animations     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

# Example Makie code (commented - requires installation)
println("\n# Example Makie Code (requires: Pkg.add(\"GLMakie\")):")
println("""
using GLMakie

# Simple interactive plot
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="y")
lines!(ax, 0..10, sin)
scatter!(ax, 1:10, rand(10))
display(fig)

# 3D surface
fig2 = Figure()
ax2 = Axis3(fig2[1, 1])
surface!(ax2, -5:0.1:5, -5:0.1:5, (x,y) -> sin(sqrt(x^2 + y^2)))
display(fig2)
""")

# ============================================================================
# SECTION 8: BEST PRACTICES FOR JULIA PROJECTS
# ============================================================================
println("\n# ================ 8. BEST PRACTICES ================")

println("""
JULIA PROJECT BEST PRACTICES:

1. PROJECT STRUCTURE
   ✓ Use Pkg.generate() to create project structure
   ✓ Keep Project.toml under version control
   ✓ Use Manifest.toml for reproducible environments
   ✓ Organize code in src/ directory
   ✓ Write tests in test/ directory

2. MODULE DESIGN
   ✓ Export only public API (use export statement)
   ✓ Prefix internal functions with underscore (_)
   ✓ Use docstrings for all public functions
   ✓ Keep modules focused and single-purpose
   ✓ Use submodules for large projects

3. TYPE DESIGN
   ✓ Use immutable structs by default (better performance)
   ✓ Use mutable struct only when necessary
   ✓ Define parametric types for flexibility
   ✓ Use abstract types for type hierarchies
   ✓ Implement Base.show() for custom types

4. PERFORMANCE
   ✓ Pre-allocate arrays when possible
   ✓ Use views (@view) instead of copies
   ✓ Avoid global variables in performance-critical code
   ✓ Use broadcasting (.+) instead of loops
   ✓ Type-stability: functions should return consistent types

5. DOCUMENTATION
   ✓ Write docstrings using Markdown
   ✓ Include examples in docstrings
   ✓ Use Documenter.jl for generating docs
   ✓ Keep README.md up to date
   ✓ Add citation information (CITATION.bib)

6. TESTING
   ✓ Use Test.jl for unit tests
   ✓ Test edge cases and error conditions
   ✓ Aim for high code coverage
   ✓ Use Continuous Integration (GitHub Actions)
   ✓ Include performance benchmarks

7. VERSION CONTROL
   ✓ Use Git for version control
   ✓ Follow semantic versioning (SemVer)
   ✓ Write meaningful commit messages
   ✓ Use branches for features/bugfixes
   ✓ Tag releases properly
""")

# ============================================================================
# SECTION 9: COMPLETE PROJECT EXAMPLE
# ============================================================================
println("\n# ================ 9. COMPLETE PROJECT EXAMPLE ================")

println("""
EXAMPLE PROJECT STRUCTURE FOR BIOLOGICAL MODELING:

BiologicalModels/
├── Project.toml
├── Manifest.toml
├── README.md
├── LICENSE
├── CITATION.bib
├── src/
│   ├── BiologicalModels.jl      # Main module file
│   ├── population/              # Population dynamics
│   │   ├── malthus.jl
│   │   ├── logistic.jl
│   │   └── lotka_volterra.jl
│   ├── epidemiology/            # Epidemiological models
│   │   ├── sir.jl
│   │   ├── seir.jl
│   │   └── parameters.jl
│   ├── physiology/              # Physiological models
│   │   ├── insulin_glucose.jl
│   │   └── immune_response.jl
│   └── utils/                   # Utility functions
│       ├── statistics.jl
│       └── visualization.jl
├── test/
│   ├── runtests.jl
│   ├── test_population.jl
│   └── test_epidemiology.jl
├── docs/
│   ├── make.jl
│   ├── Project.toml
│   └── src/
│       ├── index.md
│       ├── api.md
│       └── examples.md
├── examples/
│   ├── basic_usage.jl
│   ├── advanced_modeling.jl
│   └── real_data_analysis.jl
└── benchmarks/
    └── performance_tests.jl

PROJECT.TOML EXAMPLE:
─────────────────────
name = "BiologicalModels"
uuid = "12345678-1234-1234-1234-123456789012"
authors = ["Your Name <your.email@example.com>"]
version = "0.1.0"

[deps]
Julia = "1.9"
DifferentialEquations = "2b7a1792-96aa-5da6-8705-bd5351db8b3a"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"

[compat]
Julia = "1.9"
""")

# ============================================================================
# SECTION 10: EXERCISES
# ============================================================================
println("\n# ================ 10. EXERCISES ================")

println("""
EXERCISES FOR PART 4:

1. MODULE CREATION
   Create a module called "DataUtilities" with functions for:
   - Loading data from CSV
   - Handling missing values
   - Basic data transformations

2. CUSTOM TYPE
   Create a struct "DataSet" that contains:
   - Data (matrix or DataFrame)
   - Variable names (vector of strings)
   - Metadata (dictionary)
   Implement custom show() method

3. EXTEND STATISTICS MODULE
   Add functions to StatisticsTools:
   - Confidence interval calculation
   - Bootstrap resampling
   - Multiple comparison correction

4. VISUALIZATION
   Create a function that generates a "dashboard" plot with:
   - Histogram
   - Box plot
   - Q-Q plot
   - Summary statistics table

5. PROJECT SETUP
   Create a proper Julia project structure:
   - Initialize with Pkg
   - Add dependencies
   - Create src/ and test/ directories
   - Write basic tests

6. DOCUMENTATION
   Add comprehensive docstrings to all functions in your module
   Include examples and edge cases

7. PERFORMANCE
   Benchmark different implementations:
   - Loop vs broadcasting
   - Pre-allocated vs growing arrays
   - Views vs copies

8. INTEGRATION
   Integrate StatisticsTools with:
   - DataFrames for tabular data
   - DifferentialEquations for model output analysis
   - Plots/Makie for visualization
""")

# ============================================================================
# SECTION 11: SUMMARY
# ============================================================================
println("\n# ================ 11. SUMMARY ================")

println("""
╔══════════════════════════════════════════════════════════════════════════╗
║                         PART 4 SUMMARY                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  KEY CONCEPTS COVERED:                                                   ║
║  ─────────────────────                                                   ║
║  1. Julia project organization and structure                             ║
║  2. Module creation and organization                                     ║
║  3. Custom type definition (structs)                                     ║
║  4. Descriptive statistics implementation                                ║
║  5. Statistical visualization                                            ║
║  6. Package ecosystem overview                                           ║
║  7. Best practices for Julia development                                 ║
║                                                                          ║
║  GENERATED FILES:                                                        ║
║  ────────────────                                                        ║
║  • stats_histogram.png                                                   ║
║  • stats_boxplot.png                                                     ║
║  • stats_qq.png                                                          ║
║  • stats_distribution.png                                                ║
║  • stats_distribution_compare.png                                        ║
║                                                                          ║
║  CUSTOM MODULE:                                                          ║
║  ──────────────                                                          ║
║  • StatisticsTools - Comprehensive statistics module with:               ║
║    - StatisticalSummary type                                             ║
║    - Descriptive statistics functions                                    ║
║    - Outlier detection                                                   ║
║    - Normalization/standardization                                       ║
║    - Statistical plotting functions                                      ║
║                                                                          ║
║  NEXT STEPS (PART 5):                                                    ║
║  ────────────────────                                                    ║
║  • Applied biological modeling:                                          ║
║    - Insulin-glucose dynamics                                            ║
║    - Immune response modeling                                            ║
║    - Epidemiological models (SIR, SEIR)                                  ║
║    - Pharmacokinetic models                                              ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

println("\n=== End of Julia Basics Part 4 ===")
println("You now have the foundation to create professional Julia projects!")
println("In Part 5, we'll apply these skills to real biological modeling problems.")