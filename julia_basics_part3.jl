# Julia Basics Part 3: Plotting, Visualization, and Vector Operations
# This program introduces users to plotting, visualization, and advanced vector operations in Julia
# Topics: Plots package, vector operations, broadcasting, mutating functions, and macros

println("=== Julia Basics Part 3: Plotting, Visualization, and Vector Operations ===")
println()

# Load required packages
using Plots
using StatsPlots
using Statistics
using LinearAlgebra
using Random

# Set plot theme for better visualization
gr()  # GR backend for fast plotting

# ============================================================================
# SECTION 1: UNDERSTANDING SPECIAL JULIA SYNTAX
# ============================================================================
println("# ================ 1. SPECIAL JULIA SYNTAX ================")

# ----------------------------------------------------------------------------
# 1.1 THE BANG (!) CONVENTION - MUTATING FUNCTIONS
# ----------------------------------------------------------------------------
println("\n# --- 1.1 THE BANG (!) CONVENTION ---")
#
# In Julia, functions that modify their arguments (mutating functions) 
# conventionally end with a bang (!) character.
# This is a naming convention, not enforced by the language.
#
# Non-mutating functions return a new object without changing the original.
# Mutating functions (with !) modify the original object in place.

# Example with arrays
original_array = [1, 2, 3, 4, 5]
println("Original array: ", original_array)

# Non-mutating: sort() returns a new sorted array
sorted_array = sort(original_array)
println("After sort() - Original: ", original_array)
println("After sort() - New array: ", sorted_array)

# Mutating: sort!() modifies the original array
sort!(original_array)
println("After sort!() - Original modified: ", original_array)

# More examples of mutating vs non-mutating functions
array1 = [5, 2, 8, 1, 9]
array2 = copy(array1)

# reverse() vs reverse!()
reversed = reverse(array1)
println("\nreverse() - Original: ", array1)
println("reverse() - Result: ", reversed)

reverse!(array2)
println("reverse!() - Modified original: ", array2)

# append!() vs vcat()
arr1 = [1, 2, 3]
arr2 = [4, 5, 6]
arr3 = copy(arr1)

vcat_result = vcat(arr1, arr2)  # Non-mutating
println("\nvcat() - arr1 unchanged: ", arr1)
println("vcat() - Result: ", vcat_result)

append!(arr3, arr2)  # Mutating
println("append!() - arr3 modified: ", arr3)

# push!() adds element to end of array
numbers = [1, 2, 3]
push!(numbers, 4)
println("\nAfter push!(numbers, 4): ", numbers)

# pop!() removes and returns last element
popped = pop!(numbers)
println("After pop!(numbers) - Popped: ", popped, ", Array: ", numbers)

# ----------------------------------------------------------------------------
# 1.2 THE DOT (.) OPERATOR - BROADCASTING
# ----------------------------------------------------------------------------
println("\n# --- 1.2 THE DOT (.) OPERATOR - BROADCASTING ---")
#
# The dot (.) operator enables broadcasting, which applies operations 
# element-wise to arrays without explicit loops.
#
# This is one of Julia's most powerful features for vectorized operations.

# Basic arithmetic with broadcasting
array_a = [1, 2, 3, 4, 5]
array_b = [10, 20, 30, 40, 50]

# Without dot: would try matrix operations (may error for incompatible sizes)
# With dot: element-wise operations

println("Array A: ", array_a)
println("Array B: ", array_b)

# Element-wise addition
sum_array = array_a .+ array_b
println("\nA .+ B (element-wise): ", sum_array)

# Element-wise multiplication
product_array = array_a .* array_b
println("A .* B (element-wise): ", product_array)

# Element-wise division
division_array = array_b ./ array_a
println("B ./ A (element-wise): ", division_array)

# Element-wise exponentiation
squared_array = array_a .^ 2
println("A .^ 2 (element-wise): ", squared_array)

# Broadcasting with functions
println("\nBroadcasting with functions:")
sqrt_array = sqrt.(array_a)
println("sqrt.(A): ", sqrt_array)

log_array = log.(array_b)
println("log.(B): ", log_array)

# Broadcasting with custom functions
function add_ten(x)
    return x + 10
end

result = add_ten.(array_a)
println("\nadd_ten.(A): ", result)

# Broadcasting with multiple arguments
function multiply_add(x, y, z)
    return x * y + z
end

result_multi = multiply_add.(array_a, array_b, 5)
println("multiply_add.(A, B, 5): ", result_multi)

# Broadcasting with scalars
scalar_add = array_a .+ 100
println("\nA .+ 100 (scalar broadcasting): ", scalar_add)

scalar_mult = array_a .* 2
println("A .* 2 (scalar broadcasting): ", scalar_mult)

# Comparing loops vs broadcasting
println("\n--- Performance: Loop vs Broadcasting ---")

# Loop version
function loop_square(arr)
    result = similar(arr)
    for i in eachindex(arr)
        result[i] = arr[i]^2
    end
    return result
end

# Broadcasting version
function broadcast_square(arr)
    return arr .^ 2
end

test_array = rand(1000)
loop_result = loop_square(test_array)
broadcast_result = broadcast_square(test_array)

println("Results match: ", loop_result ≈ broadcast_result)

# ----------------------------------------------------------------------------
# 1.3 THE AT (@) SYMBOL - MACROS
# ----------------------------------------------------------------------------
println("\n# --- 1.3 THE AT (@) SYMBOL - MACROS ---")
#
# Macros in Julia are code that generates code. They are evaluated at parse time,
# before the code is executed. Macros start with @ symbol.
#
# Common macros:
# - @show: prints expression and its value
# - @time: measures execution time
# - @printf: formatted printing (from Printf package)
# - @view: creates a view instead of copying
# - @assert: assertion checking
# - @macroexpand: shows what macro expands to

# @show macro - prints expression and value
println("\n@show macro:")
x = 42
@show x
@show x^2
@show typeof(x)

# @time macro - measures execution time
println("\n@time macro:")
@time sleep(0.1)

@time begin
    sum_result = sum(1:1000000)
    println("Sum of 1 to 1,000,000: ", sum_result)
end

# @printf macro - formatted output (requires Printf package)
using Printf
println("\n@printf macro:")
@printf("Pi to 5 decimal places: %.5f\n", π)
@printf("Integer: %d, Float: %.2f, String: %s\n", 42, 3.14, "Hello")

# @view macro - creates view instead of copy (memory efficient)
println("\n@view macro:")
large_array = collect(1:1000000)

# Regular slicing creates a copy
copy_slice = large_array[1:100]
println("Type of regular slice: ", typeof(copy_slice))

# @view creates a view (reference to original data)
view_slice = @view large_array[1:100]
println("Type of view slice: ", typeof(view_slice))

# Views don't allocate new memory
println("View shares data with original: ", view_slice[1] == large_array[1])

# Modify through view affects original
view_slice[1] = 999
println("Original array[1] after view modification: ", large_array[1])

# @assert macro - assertion checking
println("\n@assert macro:")
value = 100
@assert value > 0 "Value must be positive"
println("Assertion passed: value = ", value)

# @macroexpand - shows macro expansion
println("\n@macroexpand example:")
println("@show x expands to: ")
@macroexpand @show x

# ============================================================================
# SECTION 2: INTRODUCTION TO PLOTTING WITH PLOTS.JL
# ============================================================================
println("\n# ================ 2. INTRODUCTION TO PLOTTING ================")

# Plots.jl is a plotting metapackage that provides a unified interface
# to multiple plotting backends (GR, PyPlot, PlotlyJS, etc.)

# ----------------------------------------------------------------------------
# 2.1 BASIC LINE PLOTS
# ----------------------------------------------------------------------------
println("\n# --- 2.1 BASIC LINE PLOTS ---")

# Simple line plot
x = range(0, 10, length=100)
y = sin.(x)

plot_basic = plot(x, y, 
                  linewidth=2, 
                  label="sin(x)",
                  xlabel="x",
                  ylabel="y",
                  title="Basic Line Plot",
                  legend=:topright,
                  color=:blue,
                  size=(800, 500))

savefig(plot_basic, "plot_basic_line.png")
println("Saved: plot_basic_line.png")

# Multiple lines on same plot
y2 = cos.(x)
y3 = sin.(x) .* cos.(x)

plot_multi = plot(x, y, label="sin(x)", linewidth=2, color=:blue)
plot!(x, y2, label="cos(x)", linewidth=2, color=:red)
plot!(x, y3, label="sin(x)*cos(x)", linewidth=2, color=:green)

plot!(xlabel="x", 
      ylabel="y", 
      title="Multiple Functions",
      legend=:topright,
      size=(800, 500))

savefig(plot_multi, "plot_multiple_lines.png")
println("Saved: plot_multiple_lines.png")

# ----------------------------------------------------------------------------
# 2.2 SCATTER PLOTS
# ----------------------------------------------------------------------------
println("\n# --- 2.2 SCATTER PLOTS ---")

# Generate random data
Random.seed!(42)
n_points = 50
x_scatter = rand(n_points) .* 10
y_scatter = rand(n_points) .* 10

plot_scatter = scatter(x_scatter, y_scatter,
                       markersize=8,
                       markeralpha=0.6,
                       label="Random Points",
                       xlabel="X",
                       ylabel="Y",
                       title="Scatter Plot",
                       color=:purple,
                       size=(800, 500))

savefig(plot_scatter, "plot_scatter.png")
println("Saved: plot_scatter.png")

# ----------------------------------------------------------------------------
# 2.3 BAR PLOTS
# ----------------------------------------------------------------------------
println("\n# --- 2.3 BAR PLOTS ---")

categories = ["A", "B", "C", "D", "E"]
values = [23, 45, 56, 78, 32]

plot_bar = bar(categories, values,
               label="Values",
               xlabel="Category",
               ylabel="Value",
               title="Bar Plot",
               color=:orange,
               size=(800, 500))

savefig(plot_bar, "plot_bar.png")
println("Saved: plot_bar.png")

# Grouped bar plot
values2 = [30, 35, 40, 45, 50]
plot_grouped_bar = bar(categories, [values values2],
                       label=["Group 1" "Group 2"],
                       xlabel="Category",
                       ylabel="Value",
                       title="Grouped Bar Plot",
                       bar_position=:dodge,
                       size=(800, 500))

savefig(plot_grouped_bar, "plot_grouped_bar.png")
println("Saved: plot_grouped_bar.png")

# ----------------------------------------------------------------------------
# 2.4 HISTOGRAMS
# ----------------------------------------------------------------------------
println("\n# --- 2.4 HISTOGRAMS ---")

# Generate normally distributed data
normal_data = randn(1000) .* 5 .+ 10  # mean=10, std=5

plot_histogram = histogram(normal_data,
                           bins=30,
                           label="Normal Distribution",
                           xlabel="Value",
                           ylabel="Frequency",
                           title="Histogram",
                           color=:teal,
                           alpha=0.7,
                           size=(800, 500))

savefig(plot_histogram, "plot_histogram.png")
println("Saved: plot_histogram.png")

# ----------------------------------------------------------------------------
# 2.5 SUBPLOTS
# ----------------------------------------------------------------------------
println("\n# --- 2.5 SUBPLOTS ---")

# Create multiple subplots
x_sub = range(0, 2π, length=100)

p1 = plot(x_sub, sin.(x_sub), label="sin(x)", title="Sine", color=:blue)
p2 = plot(x_sub, cos.(x_sub), label="cos(x)", title="Cosine", color=:red)
p3 = plot(x_sub, tan.(x_sub), label="tan(x)", title="Tangent", color=:green, 
          ylim=(-5, 5))
p4 = plot(x_sub, sin.(x_sub) .* cos.(x_sub), label="sin(x)*cos(x)", 
          title="Product", color=:purple)

plot_subplots = plot(p1, p2, p3, p4, 
                     layout=(2, 2),
                     size=(1000, 800),
                     xlabel="x",
                     ylabel="y")

savefig(plot_subplots, "plot_subplots.png")
println("Saved: plot_subplots.png")

# ============================================================================
# SECTION 3: ADVANCED VISUALIZATION
# ============================================================================
println("\n# ================ 3. ADVANCED VISUALIZATION ================")

# ----------------------------------------------------------------------------
# 3.1 CUSTOMIZING PLOTS
# ----------------------------------------------------------------------------
println("\n# --- 3.1 CUSTOMIZING PLOTS ---")

x_custom = range(0, 10, length=200)
y_custom = exp.(-x_custom ./ 5) .* sin.(x_custom)

plot_custom = plot(x_custom, y_custom,
                   linewidth=3,
                   linecolor=:darkblue,
                   linestyle=:solid,
                   label="Damped Sine Wave",
                   marker=:circle,
                   markersize=4,
                   markercolor=:red,
                   markeralpha=0.5,
                   xlabel="Time (s)",
                   ylabel="Amplitude",
                   title="Customized Plot with Markers",
                   titlefontsize=14,
                   legendfontsize=10,
                   guidefontsize=12,
                   tickfontsize=10,
                   grid=:both,
                   gridalpha=0.3,
                   foreground_color_legend=nothing,
                   size=(800, 500))

# Add annotations
annotate!(plot_custom, 5, 0.5, Plots.text("Peak Region", :green, 10))

# Add horizontal and vertical lines
hline!(plot_custom, [0], color=:black, linestyle=:dash, label="Zero Line")
vline!(plot_custom, [π, 2π, 3π], color=:gray, linestyle=:dot, label="Multiples of π")

savefig(plot_custom, "plot_customized.png")
println("Saved: plot_customized.png")

# ----------------------------------------------------------------------------
# 3.2 3D PLOTS
# ----------------------------------------------------------------------------
println("\n# --- 3.2 3D PLOTS ---")

# Generate 3D data
x_3d = range(-5, 5, length=50)
y_3d = range(-5, 5, length=50)
z_3d = [sin(sqrt(x^2 + y^2)) for x in x_3d, y in y_3d]

plot_3d_surface = surface(x_3d, y_3d, z_3d,
                          xlabel="X",
                          ylabel="Y",
                          zlabel="Z",
                          title="3D Surface Plot",
                          camera=(30, 60),
                          size=(800, 600))

savefig(plot_3d_surface, "plot_3d_surface.png")
println("Saved: plot_3d_surface.png")

# 3D line plot
t_3d = range(0, 4π, length=200)
x_3d_line = cos.(t_3d)
y_3d_line = sin.(t_3d)
z_3d_line = t_3d ./ (4π)

plot_3d_line = plot(x_3d_line, y_3d_line, z_3d_line,
                    xlabel="X",
                    ylabel="Y",
                    zlabel="Z",
                    title="3D Helix",
                    linewidth=3,
                    color=:rainbow,
                    size=(800, 600))

savefig(plot_3d_line, "plot_3d_helix.png")
println("Saved: plot_3d_helix.png")

# ----------------------------------------------------------------------------
# 3.3 CONTOUR PLOTS
# ----------------------------------------------------------------------------
println("\n# --- 3.3 CONTOUR PLOTS ---")

x_contour = range(-3, 3, length=100)
y_contour = range(-3, 3, length=100)
z_contour = [sin(x) * cos(y) for x in x_contour, y in y_contour]

plot_contour = contour(x_contour, y_contour, z_contour,
                       xlabel="X",
                       ylabel="Y",
                       title="Contour Plot",
                       fill=true,
                       color_palette=:viridis,
                       size=(800, 600))

savefig(plot_contour, "plot_contour.png")
println("Saved: plot_contour.png")

# ============================================================================
# SECTION 4: VECTOR AND MATRIX OPERATIONS
# ============================================================================
println("\n# ================ 4. VECTOR AND MATRIX OPERATIONS ================")

# ----------------------------------------------------------------------------
# 4.1 CREATING VECTORS AND MATRICES
# ----------------------------------------------------------------------------
println("\n# --- 4.1 CREATING VECTORS AND MATRICES ---")

# Column vectors (1D arrays)
vector1 = [1, 2, 3, 4, 5]
vector2 = collect(1:5)
vector3 = collect(range(1, 5, length=5))

println("Vector 1: ", vector1)
println("Vector 2: ", vector2)
println("Vector 3: ", vector3)

# Row vectors (using transpose)
row_vector = vector1'
println("\nRow vector (transpose): ", row_vector)
println("Type: ", typeof(row_vector))

# Matrices
matrix1 = [1 2 3; 4 5 6; 7 8 9]
matrix2 = reshape(1:9, 3, 3)

println("\nMatrix 1:")
println(matrix1)
println("\nMatrix 2:")
println(matrix2)

# Special matrices
identity_matrix = I(3)  # Identity matrix
zeros_matrix = zeros(3, 3)
ones_matrix = ones(3, 3)
random_matrix = rand(3, 3)

println("\nIdentity matrix:")
println(identity_matrix)
println("\nZeros matrix:")
println(zeros_matrix)

# ----------------------------------------------------------------------------
# 4.2 MATRIX OPERATIONS
# ----------------------------------------------------------------------------
println("\n# --- 4.2 MATRIX OPERATIONS ---")

A = [1 2; 3 4]
B = [5 6; 7 8]

println("Matrix A:")
println(A)
println("\nMatrix B:")
println(B)

# Matrix addition
C = A + B
println("\nA + B:")
println(C)

# Matrix multiplication
D = A * B
println("\nA * B (matrix multiplication):")
println(D)

# Element-wise multiplication
E = A .* B
println("\nA .* B (element-wise):")
println(E)

# Matrix transpose
println("\nA' (transpose):")
println(A')

# Matrix inverse
A_inv = inv(A)
println("\nA^(-1) (inverse):")
println(A_inv)

# Verify inverse
identity_check = A * A_inv
println("\nA * A^(-1) (should be identity):")
println(identity_check)

# Determinant
det_A = det(A)
println("\ndet(A) = ", det_A)

# Eigenvalues and eigenvectors
eigen_decomp = eigen(A)
println("\nEigenvalues: ", eigen_decomp.values)
println("Eigenvectors:")
println(eigen_decomp.vectors)

# ----------------------------------------------------------------------------
# 4.3 LINEAR ALGEBRA OPERATIONS
# ----------------------------------------------------------------------------
println("\n# --- 4.3 LINEAR ALGEBRA OPERATIONS ---")

# Dot product
v1 = [1, 2, 3]
v2 = [4, 5, 6]
dot_product = dot(v1, v2)
println("Dot product of ", v1, " and ", v2, " = ", dot_product)

# Cross product (3D vectors only)
cross_product = cross(v1, v2)
println("Cross product: ", cross_product)

# Norm (magnitude)
norm_v1 = norm(v1)
println("Norm of v1: ", norm_v1)

# Normalize vector
normalized_v1 = v1 / norm(v1)
println("Normalized v1: ", normalized_v1)
println("Norm of normalized: ", norm(normalized_v1))

# Solving linear systems: Ax = b
A_sys = [2 1; 1 3]
b_sys = [5, 6]
x_solution = A_sys \ b_sys  # Backslash operator solves linear systems
println("\nSolving Ax = b:")
println("A = ", A_sys)
println("b = ", b_sys)
println("x = ", x_solution)

# Verify solution
verification = A_sys * x_solution
println("Verification (A*x): ", verification)

# ----------------------------------------------------------------------------
# 4.4 ARRAY COMPREHENSIONS AND GENERATORS
# ----------------------------------------------------------------------------
println("\n# --- 4.4 ARRAY COMPREHENSIONS ---")

# Basic comprehension
squares = [x^2 for x in 1:10]
println("Squares 1-10: ", squares)

# Comprehension with condition
even_squares = [x^2 for x in 1:10 if x % 2 == 0]
println("Even squares 1-10: ", even_squares)

# Multi-dimensional comprehension
multiplication_table = [i * j for i in 1:5, j in 1:5]
println("\nMultiplication table (5x5):")
println(multiplication_table)

# Generator expression (lazy evaluation)
generator = (x^2 for x in 1:10)
println("\nGenerator type: ", typeof(generator))
println("First 5 values from generator: ", collect(generator)[1:5])

# ============================================================================
# SECTION 5: PRACTICAL EXAMPLES
# ============================================================================
println("\n# ================ 5. PRACTICAL EXAMPLES ================")

# ----------------------------------------------------------------------------
# 5.1 DATA ANALYSIS VISUALIZATION
# ----------------------------------------------------------------------------
println("\n# --- 5.1 DATA ANALYSIS VISUALIZATION ---")

# Generate synthetic experimental data
Random.seed!(123)
n_samples = 100
time_data = range(0, 10, length=n_samples)
signal_data = sin.(time_data) .+ 0.3 .* randn(n_samples)  # Signal with noise

# Plot raw data
plot_raw = scatter(time_data, signal_data,
                   markersize=4,
                   markeralpha=0.5,
                   label="Raw Data",
                   color=:lightblue,
                   xlabel="Time",
                   ylabel="Signal",
                   title="Experimental Data with Noise",
                   size=(800, 500))

# Add smoothed line
smoothed_data = [mean(signal_data[max(1,i-2):min(n_samples,i+2)]) for i in 1:n_samples]
plot!(plot_raw, time_data, smoothed_data,
      linewidth=2,
      label="Smoothed",
      color=:red)

savefig(plot_raw, "plot_data_analysis.png")
println("Saved: plot_data_analysis.png")

# ----------------------------------------------------------------------------
# 5.2 FUNCTION COMPARISON
# ----------------------------------------------------------------------------
println("\n# --- 5.2 FUNCTION COMPARISON ---")

x_compare = range(0.1, 10, length=200)

# Compare different growth functions
linear = x_compare
quadratic = x_compare .^ 2
logarithmic = log.(x_compare)
exponential = exp.(x_compare ./ 5)

plot_compare = plot(x_compare, linear, label="Linear (x)", linewidth=2, color=:blue)
plot!(x_compare, quadratic, label="Quadratic (x²)", linewidth=2, color=:red)
plot!(x_compare, logarithmic, label="Logarithmic (ln(x))", linewidth=2, color=:green)
plot!(x_compare, exponential, label="Exponential (e^(x/5))", linewidth=2, color=:purple)

plot!(xlabel="x",
      ylabel="f(x)",
      title="Comparison of Growth Functions",
      legend=:topleft,
      grid=:both,
      gridalpha=0.3,
      size=(800, 500))

savefig(plot_compare, "plot_function_comparison.png")
println("Saved: plot_function_comparison.png")

# ----------------------------------------------------------------------------
# 5.3 STATISTICAL VISUALIZATION
# ----------------------------------------------------------------------------
println("\n# --- 5.3 STATISTICAL VISUALIZATION ---")

# Generate sample data for different groups
Random.seed!(456)
group_a = randn(100) .* 2 .+ 10
group_b = randn(100) .* 2 .+ 12
group_c = randn(100) .* 2 .+ 11

# Box plot
plot_box = boxplot(["Group A", "Group B", "Group C"],
                   [group_a, group_b, group_c],  # Запятые вместо пробелов!
                   label="",
                   ylabel="Value",
                   title="Box Plot Comparison",
                   color=[:blue :red :green],
                   size=(800, 500))
                   
savefig(plot_box, "plot_boxplot.png")
println("Saved: plot_boxplot.png")

# Calculate and display statistics
println("\nStatistical Summary:")
println("Group A - Mean: ", round(mean(group_a), digits=2), 
        ", Std: ", round(std(group_a), digits=2))
println("Group B - Mean: ", round(mean(group_b), digits=2), 
        ", Std: ", round(std(group_b), digits=2))
println("Group C - Mean: ", round(mean(group_c), digits=2), 
        ", Std: ", round(std(group_c), digits=2))

# ============================================================================
# SECTION 6: PERFORMANCE TIPS
# ============================================================================
println("\n# ================ 6. PERFORMANCE TIPS ================")

# ----------------------------------------------------------------------------
# 6.1 PRE-ALLOCATION
# ----------------------------------------------------------------------------
println("\n# --- 6.1 PRE-ALLOCATION ---")

n = 10000

# Bad: growing array in loop (slow)
function bad_growth(n)
    result = []
    for i in 1:n
        push!(result, i^2)
    end
    return result
end

# Good: pre-allocate array (fast)
function good_growth(n)
    result = zeros(Int64, n)
    for i in 1:n
        result[i] = i^2
    end
    return result
end

# Test both
@time bad_result = bad_growth(n)
@time good_result = good_growth(n)

println("Results match: ", bad_result == good_result)

# ----------------------------------------------------------------------------
# 6.2 VIEWS VS COPIES
# ----------------------------------------------------------------------------
println("\n# --- 6.2 VIEWS VS COPIES ---")

large_data = rand(10000, 100)

# Copy (allocates new memory)
@time copy_data = large_data[:, 1:50]

# View (no allocation)
@time view_data = @view large_data[:, 1:50]

println("Copy type: ", typeof(copy_data))
println("View type: ", typeof(view_data))

# ----------------------------------------------------------------------------
# 6.3 BROADCASTING VS LOOPS
# ----------------------------------------------------------------------------
println("\n# --- 6.3 BROADCASTING VS LOOPS ---")

test_data = rand(10000)

# Loop version
function loop_version(arr)
    result = similar(arr)
    for i in eachindex(arr)
        result[i] = sqrt(arr[i]) + log(arr[i] + 1)
    end
    return result
end

# Broadcasting version
function broadcast_version(arr)
    return sqrt.(arr) .+ log.(arr .+ 1)
end

@time loop_result = loop_version(test_data)
@time broadcast_result = broadcast_version(test_data)

println("Results match: ", loop_result ≈ broadcast_result)

# ============================================================================
# SECTION 7: SUMMARY AND EXERCISES
# ============================================================================
println("\n# ================ 7. SUMMARY AND EXERCISES ================")

println("""
SUMMARY OF PART 3:

Key Concepts Covered:
1. Bang (!) convention for mutating functions
2. Dot (.) operator for broadcasting
3. At (@) symbol for macros
4. Basic and advanced plotting with Plots.jl
5. Vector and matrix operations
6. Linear algebra with LinearAlgebra
7. Performance optimization tips

Generated Plot Files:
- plot_basic_line.png
- plot_multiple_lines.png
- plot_scatter.png
- plot_bar.png
- plot_grouped_bar.png
- plot_histogram.png
- plot_subplots.png
- plot_customized.png
- plot_3d_surface.png
- plot_3d_helix.png
- plot_contour.png
- plot_data_analysis.png
- plot_function_comparison.png
- plot_boxplot.png

EXERCISES FOR PRACTICE:

1. Create a plot showing the trajectory of a projectile with different 
   initial angles (30°, 45°, 60°).

2. Use broadcasting to create a multiplication table from 1 to 10.

3. Create a function that normalizes any vector to unit length, then 
   test it with random vectors.

4. Generate 1000 random points and create a 2D histogram showing their 
   density distribution.

5. Compare the performance of loop-based vs broadcasting-based 
   implementation for matrix multiplication.

6. Create an animated plot showing a sine wave moving over time.

7. Use @view to create multiple views of a large array and modify them 
   to see how changes affect the original.

8. Create a dashboard with 4 subplots showing different statistical 
   distributions (normal, uniform, exponential, binomial).
""")

println("\n=== End of Julia Basics Part 3 ===")
println("In Part 4, we will apply these concepts to biological modeling:")
println("- Insulin-glucose dynamics")
println("- Immune response modeling")
println("- Epidemiological models (SIR, SEIR)")
println("- Pharmacokinetic models")