# Julia Basics Part 2: Collections and Advanced Types
# This program introduces users to Julia's collection types and advanced type system
# Topics covered: Arrays, Tuples, Dictionaries, Sets, Type System, and Structs

println("=== Julia Basics Part 2: Collections and Advanced Types ===")
println()

# ================ 1. ARRAYS ================
println("# ================ 1. ARRAYS ================")
# Arrays are fundamental data structures in Julia
# Julia arrays are 1-indexed (first element is at index 1)

# Creating arrays
int_array = [1, 2, 3, 4, 5]
float_array = [1.0, 2.0, 3.0, 4.0, 5.0]
mixed_array = [1, 2.0, "three", 4]  # Will be converted to Any type

println("Integer array: ", int_array)
println("Float array: ", float_array)
println("Mixed array: ", mixed_array)
println("Type of mixed array: ", typeof(mixed_array))

# Array with specific type
typed_array::Array{Int64, 1} = [10, 20, 30, 40, 50]
println("\nTyped array: ", typed_array)
println("Element type: ", eltype(typed_array))
println("Number of dimensions: ", ndims(typed_array))
println("Length: ", length(typed_array))

# Multi-dimensional arrays
matrix_2x3 = [1 2 3; 4 5 6]  # 2 rows, 3 columns
println("\n2x3 Matrix:")
println(matrix_2x3)
println("Size: ", size(matrix_2x3))

matrix_3x3 = [1 2 3; 4 5 6; 7 8 9]
println("\n3x3 Matrix:")
println(matrix_3x3)

# Array comprehension
squares = [x^2 for x in 1:10]
println("\nSquares (1-10) using comprehension: ", squares)

# Array comprehension with condition
even_squares = [x^2 for x in 1:10 if x % 2 == 0]
println("Even squares (1-10): ", even_squares)

# Creating arrays with functions
zeros_array = zeros(5)  # Array of 5 zeros
ones_array = ones(3, 3)  # 3x3 array of ones
range_array = collect(1:2:10)  # Range from 1 to 10 with step 2

println("\nZeros array: ", zeros_array)
println("Ones array:\n", ones_array)
println("Range array: ", range_array)

# Accessing array elements
println("\nAccessing array elements:")
println("First element: ", int_array[1])
println("Last element: ", int_array[end])
println("Elements 2-4: ", int_array[2:4])
println("Every second element: ", int_array[1:2:end])

# Modifying arrays
int_array[1] = 100  # Change first element
println("\nAfter modifying first element: ", int_array)

push!(int_array, 6)  # Add element to end
println("After push!: ", int_array)

pop!(int_array)  # Remove last element
println("After pop!: ", int_array)

insert!(int_array, 1, 0)  # Insert at position 1
println("After insert!: ", int_array)

deleteat!(int_array, 1)  # Delete at position 1
println("After deleteat!: ", int_array)

# Array operations
array1 = [1, 2, 3]
array2 = [4, 5, 6]
concatenated = vcat(array1, array2)  # Vertical concatenation
println("\nConcatenated arrays: ", concatenated)

# Matrix operations
matrix_a = [1 2; 3 4]
matrix_b = [5 6; 7 8]
matrix_sum = matrix_a + matrix_b
matrix_product = matrix_a * matrix_b
println("\nMatrix A:\n", matrix_a)
println("Matrix B:\n", matrix_b)
println("A + B:\n", matrix_sum)
println("A * B:\n", matrix_product)

# ================ 2. TUPLES ================
println("\n# ================ 2. TUPLES ================")
# Tuples are immutable ordered collections
# Created with parentheses or comma-separated values

# Creating tuples
tuple1 = (1, 2, 3)
tuple2 = "a", "b", "c"  # Parentheses are optional
single_element_tuple = (42,)  # Note the comma for single element

println("Tuple 1: ", tuple1)
println("Tuple 2: ", tuple2)
println("Single element tuple: ", single_element_tuple)
println("Type: ", typeof(tuple1))

# Accessing tuple elements
println("\nAccessing tuple elements:")
println("First element: ", tuple1[1])
println("Last element: ", tuple1[end])

# Tuple unpacking
x, y, z = tuple1
println("\nUnpacked values: x=$x, y=$y, z=$z")

# Named tuples
person = (name="Alice", age=30, city="New York")
println("\nNamed tuple: ", person)
println("Name: ", person.name)
println("Age: ", person.age)

# Tuples are immutable (cannot be modified)
# tuple1[1] = 100  # This would cause an error!

# ================ 3. DICTIONARIES ================
println("\n# ================ 3. DICTIONARIES ================")
# Dictionaries are key-value pairs
# Created with Dict() or dictionary literal syntax

# Creating dictionaries
dict1 = Dict("name" => "Bob", "age" => 25, "city" => "London")
dict2 = Dict(
    "apple" => 1.50,
    "banana" => 0.75,
    "orange" => 2.00
)

println("Dictionary 1: ", dict1)
println("Dictionary 2: ", dict2)

# Accessing dictionary values
println("\nAccessing dictionary values:")
println("Name: ", dict1["name"])
println("Age: ", dict1["age"])

# Safe access with get() (returns default if key doesn't exist)
country = get(dict1, "country", "Unknown")
println("Country (with default): ", country)

# Adding and modifying entries
dict1["email"] = "bob@example.com"
println("\nAfter adding email: ", dict1)

dict1["age"] = 26
println("After updating age: ", dict1)

# Removing entries
delete!(dict1, "city")
println("After deleting city: ", dict1)

# Dictionary operations
println("\nDictionary operations:")
println("Keys: ", keys(dict1))
println("Values: ", values(dict1))
println("Has key 'name'? ", haskey(dict1, "name"))
println("Number of pairs: ", length(dict1))

# Iterating over dictionary
println("\nIterating over dictionary:")
for (key, value) in dict1
    println("$key: $value")
end

# ================ 4. SETS ================
println("\n# ================ 4. SETS ================")
# Sets are unordered collections of unique elements
# Created with Set()

# Creating sets
set1 = Set([1, 2, 3, 4, 5])
set2 = Set([3, 4, 5, 6, 7])

println("Set 1: ", set1)
println("Set 2: ", set2)

# Set operations
union_set = union(set1, set2)  # All elements from both sets
intersect_set = intersect(set1, set2)  # Common elements
diff_set = setdiff(set1, set2)  # Elements in set1 but not in set2

println("\nUnion: ", union_set)
println("Intersection: ", intersect_set)
println("Difference (set1 - set2): ", diff_set)

# Set membership
println("\nSet membership:")
println("3 in set1? ", 3 in set1)
println("8 in set1? ", 8 in set1)

# Adding and removing elements
push!(set1, 6)
println("\nAfter adding 6: ", set1)

pop!(set1)  # Remove an arbitrary element
println("After pop!: ", set1)

# ================ 5. TYPE SYSTEM ================
println("\n# ================ 5. TYPE SYSTEM ================")
# Julia has a rich type hierarchy
# All types are subtypes of Any

# Checking types
value1 = 42
value2 = 3.14
value3 = "Hello"

println("Type of 42: ", typeof(value1))
println("Type of 3.14: ", typeof(value2))
println("Type of 'Hello': ", typeof(value3))

# Type hierarchy
println("\nType hierarchy:")
println("Int64 <: Integer? ", Int64 <: Integer)
println("Integer <: Real? ", Integer <: Real)
println("Real <: Number? ", Real <: Number)
println("Number <: Any? ", Number <: Any)

# Type checking
println("\nType checking:")
println("Is 42 an Integer? ", isa(value1, Integer))
println("Is 3.14 a Real? ", isa(value2, Real))
println("Is 'Hello' a String? ", isa(value3, String))

# Type conversion
println("\nType conversion:")
int_to_float = Float64(42)
float_to_int = Int64( round(3.99) )  # Truncates, doesn't round
string_to_int = parse(Int64, "123")

println("Int to Float: ", int_to_float, " (type: ", typeof(int_to_float), ")")
println("Float to Int: ", float_to_int, " (type: ", typeof(float_to_int), ")")
println("String to Int: ", string_to_int, " (type: ", typeof(string_to_int), ")")

# ================ 6. CUSTOM TYPES (STRUCTS) ================
println("\n# ================ 6. CUSTOM TYPES (STRUCTS) ================")
# Structs allow you to define custom data types

# Basic struct definition
struct Person
    name::String
    age::Int
    email::String
end

# Creating struct instances
person1 = Person("Alice", 30, "alice@example.com")
person2 = Person("Bob", 25, "bob@example.com")

println("Person 1: ", person1)
println("Person 2: ", person2)

# Accessing struct fields
println("\nAccessing struct fields:")
println("Person 1 name: ", person1.name)
println("Person 1 age: ", person1.age)
println("Person 1 email: ", person1.email)

# Structs are immutable by default
# person1.age = 31  # This would cause an error!

# Mutable struct (can be modified after creation)
mutable struct MutablePerson
    name::String
    age::Int
    email::String
end

mutable_person = MutablePerson("Charlie", 35, "charlie@example.com")
println("\nMutable person: ", mutable_person)

mutable_person.age = 36  # This works!
println("After updating age: ", mutable_person)

# Struct with default values (using inner constructor)
struct Point
    x::Float64
    y::Float64
    Point(x, y) = new(x, y)
    Point() = new(0.0, 0.0)  # Default constructor
end

point1 = Point()
point2 = Point(3.0, 4.0)

println("\nPoint with defaults: ", point1)
println("Point with values: ", point2)

# ================ 7. PARAMETRIC TYPES ================
println("\n# ================ 7. PARAMETRIC TYPES ================")
# Parametric types allow type parameters

# Parametric struct
struct Container{T}
    value::T
end

int_container = Container{Int}(42)
float_container = Container{Float64}(3.14)
string_container = Container{String}("Hello")

println("Int container: ", int_container)
println("Float container: ", float_container)
println("String container: ", string_container)

# Type parameter can be inferred
inferred_container = Container("World")
println("Inferred container: ", inferred_container)
println("Type: ", typeof(inferred_container))

# ================ 8. UNION TYPES ================
println("\n# ================ 8. UNION TYPES ================")
# Union types allow a variable to be one of several types

# Creating union type
StringOrInt = Union{String, Int}

function process_value(value::StringOrInt)
    if isa(value, String)
        return "String: $value"
    else
        return "Integer: $value"
    end
end

println(process_value("Hello"))
println(process_value(42))

# Nothing type (similar to null in other languages)
maybe_value::Union{String, Nothing} = nothing
println("\nMaybe value: ", maybe_value)

maybe_value = "Now has a value"
println("Maybe value after assignment: ", maybe_value)

# ================ 9. TYPE ALIASES ================
println("\n# ================ 9. TYPE ALIASES ================")
# Type aliases make complex types easier to read

const Point2D = Tuple{Float64, Float64}
const StringDict = Dict{String, Any}

point_2d::Point2D = (1.0, 2.0)
string_dict::StringDict = Dict("key1" => "value1", "key2" => 42)

println("Point2D: ", point_2d)
println("StringDict: ", string_dict)

# ================ 10. ABSTRACT TYPES ================
println("\n# ================ 10. ABSTRACT TYPES ================")
# Abstract types cannot be instantiated but can have subtypes

abstract type Shape end
abstract type TwoDShape <: Shape end
abstract type ThreeDShape <: Shape end

struct Circle <: TwoDShape
    radius::Float64
end

struct Rectangle <: TwoDShape
    width::Float64
    height::Float64
end

struct Sphere <: ThreeDShape
    radius::Float64
end

circle = Circle(5.0)
rectangle = Rectangle(4.0, 6.0)
sphere = Sphere(3.0)

println("Circle: ", circle)
println("Rectangle: ", rectangle)
println("Sphere: ", sphere)

# Type checking with abstract types
println("\nType checking:")
println("Circle is a Shape? ", circle isa Shape)
println("Circle is a TwoDShape? ", circle isa TwoDShape)
println("Sphere is a TwoDShape? ", sphere isa TwoDShape)

# Function that works with any Shape
function describe_shape(shape::Shape)
    return "This is a $(typeof(shape))"
end

println("\nDescribe circle: ", describe_shape(circle))
println("Describe rectangle: ", describe_shape(rectangle))
println("Describe sphere: ", describe_shape(sphere))

# ================ 11. NESTED COLLECTIONS ================
println("\n# ================ 11. NESTED COLLECTIONS ================")
# Collections can contain other collections

# Array of arrays
array_of_arrays = [[1, 2], [3, 4], [5, 6]]
println("Array of arrays: ", array_of_arrays)
println("First inner array: ", array_of_arrays[1])
println("First element of first inner array: ", array_of_arrays[1][1])

# Dictionary with array values
inventory = Dict(
    "fruits" => ["apple", "banana", "orange"],
    "vegetables" => ["carrot", "broccoli", "spinach"],
    "grains" => ["rice", "wheat", "oats"]
)

println("\nInventory: ", inventory)
println("Fruits: ", inventory["fruits"])

# Array of dictionaries
users = [
    Dict("name" => "Alice", "age" => 30),
    Dict("name" => "Bob", "age" => 25),
    Dict("name" => "Charlie", "age" => 35)
]

println("\nUsers:")
for (i, user) in enumerate(users)
    println("User $i: $(user["name"]), Age: $(user["age"])")
end

# ================ 12. SLICING AND INDEXING ================
println("\n# ================ 12. SLICING AND INDEXING ================")
# Advanced array slicing and indexing

data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

println("Original data: ", data)
println("First 3 elements: ", data[1:3])
println("Last 3 elements: ", data[end-2:end])
println("Every 2nd element: ", data[1:2:end])
println("Reverse: ", data[end:-1:1])

# Boolean indexing
mask = data .> 50  # Create boolean mask
println("\nElements > 50: ", data[mask])

# Finding indices
indices = findall(x -> x > 50, data)
println("Indices of elements > 50: ", indices)

# ================ 13. BROADCASTING ================
println("\n# ================ 13. BROADCASTING ================")
# Broadcasting applies operations element-wise

array_a = [1, 2, 3, 4, 5]
array_b = [10, 20, 30, 40, 50]

# Element-wise operations (note the dot)
sum_array = array_a .+ array_b
product_array = array_a .* array_b
squared_array = array_a .^ 2

println("Array A: ", array_a)
println("Array B: ", array_b)
println("A .+ B: ", sum_array)
println("A .* B: ", product_array)
println("A .^ 2: ", squared_array)

# Broadcasting with functions
sqrt_array = sqrt.(array_a)
println("sqrt.(A): ", sqrt_array)

# ================ 14. MEMORY AND PERFORMANCE TIPS ================
println("\n# ================ 14. MEMORY AND PERFORMANCE TIPS ================")
# Pre-allocate arrays when possible
println("Pre-allocation example:")
n = 1000000

# Bad: growing array in loop (slow)
# result_bad = []
# for i in 1:n
#     push!(result_bad, i^2)
# end

# Good: pre-allocate array (fast)
result_good = zeros(Int64, n)
for i in 1:n
    result_good[i] = i^2
end

println("Pre-allocated array length: ", length(result_good))
println("First 10 elements: ", result_good[1:10])

# Use views instead of copies when possible
large_array = collect(1:100)
view_array = @view large_array[1:10]  # Creates a view, not a copy
copy_array = large_array[1:10]  # Creates a copy

println("\nView type: ", typeof(view_array))
println("Copy type: ", typeof(copy_array))

println("\n=== End of Julia Basics Part 2 ===")
println("In the next parts, we'll cover plotting, visualization, external libraries, and module creation.")