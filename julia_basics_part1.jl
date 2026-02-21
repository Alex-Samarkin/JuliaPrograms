# Julia Basics Part 1: Introduction to the Language
# This program introduces users to the fundamentals of the Julia programming language
# Topics covered: Basic syntax, data types, variables, input/output, control flow, and functions

println("=== Julia Basics Part 1: Introduction to the Language ===")
println()

# ================ 1. VARIABLES AND ASSIGNMENT ================
println("# ================ 1. VARIABLES AND ASSIGNMENT ================")

# Julia uses dynamic typing - variables are assigned values without explicit type declaration
name = "Julia"
version = 1.8
release_year = 2012
is_popular = true

println("Language: ", name)
println("Version: ", version)
println("Release Year: ", release_year)
println("Is Popular: ", is_popular)

# Variables can change type during runtime (dynamic typing)
println("\nDemonstrating dynamic typing:")
println("Initial type of 'version': ", typeof(version))
version = "1.9.0"  # Now it's a string
println("New type of 'version': ", typeof(version))

# Naming conventions: use snake_case for variables and functions
user_count = 100
total_amount = 123.45

# Constants are declared with const and should be UPPERCASE
const PI = 3.141592653589793
const MAX_USERS = 1000

println("\nConstants:")
println("PI = ", PI)
println("MAX_USERS = ", MAX_USERS)

# ================ 2. BASIC DATA TYPES ================
println("\n# ================ 2. BASIC DATA TYPES ================")

# Integers come in various sizes
int8_var::Int8 = 127      # 8-bit signed integer (-128 to 127)
int16_var::Int16 = 32767  # 16-bit signed integer
int32_var::Int32 = 2147483647  # 32-bit signed integer
int64_var::Int64 = 9223372036854775807  # 64-bit signed integer

println("Integer types:")
println("Int8: ", int8_var, " (type: ", typeof(int8_var), ")")
println("Int16: ", int16_var, " (type: ", typeof(int16_var), ")")
println("Int32: ", int32_var, " (type: ", typeof(int32_var), ")")
println("Int64: ", int64_var, " (type: ", typeof(int64_var), ")")

# Unsigned integers
uint8_var::UInt8 = 0xff  # Hexadecimal notation
uint32_var::UInt32 = 0xffffffff

println("\nUnsigned integers:")
println("UInt8: ", uint8_var, " (hex: 0x", string(uint8_var, base=16), ")")
println("UInt32: ", uint32_var, " (hex: 0x", string(uint32_var, base=16), ")")

# Floating-point numbers
float32_var::Float32 = 3.14159f0
float64_var::Float64 = 3.141592653589793

println("\nFloating-point types:")
println("Float32: ", float32_var, " (type: ", typeof(float32_var), ")")
println("Float64: ", float64_var, " (type: ", typeof(float64_var), ")")

# Scientific notation
scientific_num = 1.23e-4
println("Scientific notation: ", scientific_num)

# Complex numbers
complex_num = 3 + 4im
println("\nComplex number: ", complex_num)
println("Real part: ", real(complex_num))
println("Imaginary part: ", imag(complex_num))
println("Magnitude: ", abs(complex_num))
println("Conjugate: ", conj(complex_num))

# Rational numbers
rational_num = 22//7
println("\nRational number: ", rational_num)
println("As float: ", Float64(rational_num))

# Boolean type
bool_true = true
bool_false = false
println("\nBoolean values: ", bool_true, " and ", bool_false)

# Characters and strings
char_a = 'A'
string_hello = "Hello, Julia!"
println("\nCharacter: ", char_a, " (type: ", typeof(char_a), ")")
println("String: ", string_hello, " (type: ", typeof(string_hello), ")")

# ================ 3. STRINGS AND STRING OPERATIONS ================
println("\n# ================ 3. STRINGS AND STRING OPERATIONS ================")

# String concatenation
first_name = "John"
last_name = "Doe"
full_name = first_name * " " * last_name
println("Full name: ", full_name)

# String interpolation (preferred method)
age = 30
info = "Name: $full_name, Age: $age"
println("Interpolated string: ", info)

# Multi-line strings
multiline_string = """
    This is a multi-line string.
    It can span multiple lines.
    Very useful for formatted text.
"""
println("Multi-line string:\n", multiline_string)

# String operations
text = "Julia Programming Language"
println("Original text: ", text)
println("Length: ", length(text))
println("Uppercase: ", uppercase(text))
println("Lowercase: ", lowercase(text))
println("Title case: ", titlecase(text))
println("Contains 'Program'? ", occursin("Program", text))
println("Replace 'Language' with 'System': ", replace(text, "Language" => "System"))

# Accessing characters and substrings
println("First character: ", text[1])
println("Last character: ", text[end])
println("Substring (1:5): ", text[1:5])
println("Last 8 characters: ", text[(end-7):end])

# ================ 4. INPUT/OUTPUT OPERATIONS ================
println("\n# ================ 4. INPUT/OUTPUT OPERATIONS ================")

# Basic output with println (with newline) and print (without newline)
print("This is printed without a newline")
print(" so this continues on the same line\n")

# Formatted output
# for using @sprintf
using Printf
value = 42
formatted_output = @sprintf("The answer is %d", value)
println(formatted_output)

# Reading from standard input (commented out to avoid blocking execution)
# println("Enter your name: ")
# user_input = readline()
# println("Hello, $user_input!")

# Reading numbers from input
# println("Enter a number: ")
# number_input = parse(Float64, readline())
# println("You entered: $number_input, squared: $(number_input^2)")

# File operations - writing
println("\nFile operations:")
filename = "sample.txt"
open(filename, "w") do file
    write(file, "Hello from Julia!\n")
    write(file, "This is a sample file.\n")
    write(file, "Created for learning purposes.\n")
end
println("Written content to $filename")

# File operations - reading
content = open(filename, "r") do file
    read(file, String)
end
println("Content read from $filename:")
println(content)

# Check if file exists
println("Does '$filename' exist? ", isfile(filename))

# Clean up - remove the sample file
rm(filename)
println("Removed $filename")

# ================ 5. CONTROL FLOW STATEMENTS ================
println("\n# ================ 5. CONTROL FLOW STATEMENTS ================")

# Conditional statements
temperature = 25
if temperature > 30
    println("It's hot outside!")
elseif temperature > 20
    println("It's warm outside! Temperature: $temperature °C")
else
    println("It's cool outside! Temperature: $temperature °C")
end

# Ternary operator
weather = temperature > 20 ? "warm" : "cool"
println("Weather condition: $weather")

# Short-circuit evaluation
is_sunny = true
is_warm = temperature > 20
should_go_outside = is_sunny && is_warm  # Only evaluates second condition if first is true
println("Should go outside? $should_go_outside")

# Loops - for loops
println("\nFor loop example:")
for i in 1:5
    println("Iteration $i")
end

# For loop with arrays
fruits = ["apple", "banana", "orange"]
println("\nFruits list:")
for (index, fruit) in enumerate(fruits)
    println("$index. $fruit")
end

# While loop
println("\nWhile loop example:")
counter = 1
while counter <= 3
    println("Counter: $counter")
    counter += 1
end

# Loop control - break and continue
println("\nLoop with break and continue:")
for i in 1:10
    if i == 3
        continue  # Skip the rest of this iteration
    end
    if i == 7
        break  # Exit the loop
    end
    println("Number: $i")
end

# Exception handling
println("\nException handling example:")
try
    result = 10 / 0  # This would cause an error
catch e
    println("An error occurred: ", e)
finally
    println("This runs regardless of whether an exception occurred")
end

# Safe division example
function safe_divide(a, b)
    try
        return a / b
    catch e
        println("Error in division: ", e)
        return nothing
    end
end

println("Safe division 10/2: ", safe_divide(10, 2))
println("Safe division 10/0: ", safe_divide(10, 0))

# ================ 6. FUNCTIONS ================
println("\n# ================ 6. FUNCTIONS ================")

# Basic function definition
function greet(name)
    return "Hello, $name !"
end

println(greet("Alice"))

# Compact function syntax
square(x) = x^2
println("Square of 5: ", square(5))

# Functions with multiple arguments
function calculate_area(length, width)
    return length * width
end

area = calculate_area(10, 5)
println("Area of rectangle (10x5): $area")

# Functions with default arguments
function introduce(name, age=25, city="Unknown")
    return "Hi, I'm $name, $age years old, from $city."
end

println(introduce("Bob"))
println(introduce("Charlie", 30))
println(introduce("Diana", 28, "New York"))

# Functions with keyword arguments
function create_profile(; name, age, email="unknown@email.com", phone=nothing)
    profile = Dict(
        "name" => name,
        "age" => age,
        "email" => email,
        "phone" => phone
    )
    return profile
end

profile = create_profile(name="Eve", age=35, email="eve@example.com")
println("Profile: ", profile)

# Anonymous functions (lambda functions)
multiply_by_two = x -> x * 2
println("Anonymous function result: ", multiply_by_two(7))

# Higher-order functions example
numbers = [1, 2, 3, 4, 5]
squared_numbers = map(square, numbers)  # Apply square function to each element
println("Original numbers: ", numbers)
println("Squared numbers: ", squared_numbers)

# Filter function
even_numbers = filter(x -> x % 2 == 0, numbers)
println("Even numbers: ", even_numbers)

# Reduce function
sum_of_numbers = reduce(+, numbers)
println("Sum of numbers: ", sum_of_numbers)

# Multiple return values
function divide_with_remainder(dividend, divisor)
    quotient = dividend ÷ divisor  # Integer division
    remainder = dividend % divisor  # Modulo operation
    return quotient, remainder
end

quot, rem = divide_with_remainder(17, 5)
println("17 ÷ 5 = $quot with remainder $rem")

println("\n=== End of Julia Basics Part 1 ===")
println("In the next parts, we'll cover collections, advanced types, plotting, external libraries, and module creation.")