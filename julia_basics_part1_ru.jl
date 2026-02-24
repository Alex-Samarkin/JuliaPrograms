# Основы Julia Часть 1: Введение в язык
# Эта программа знакомит пользователей с основами языка программирования Julia
# Темы: Базовый синтаксис, типы данных, переменные, ввод/вывод, управляющие конструкции и функции

println("=== Основы Julia Часть 1: Введение в язык ===")
println()

# ================ 1. ПЕРЕМЕННЫЕ И ПРИСВАИВАНИЕ ================
println("# ================ 1. ПЕРЕМЕННЫЕ И ПРИСВАИВАНИЕ ================")
# Julia использует динамическую типизацию - переменным присваиваются значения без явного объявления типа
name = "Julia"
version = 1.8
release_year = 2012
is_popular = true

println("Язык: ", name)
println("Версия: ", version)
println("Год выпуска: ", release_year)
println("Популярен: ", is_popular)

# Переменные могут менять тип во время выполнения (динамическая типизация)
println("\nДемонстрация динамической типизации:")
println("Начальный тип 'version': ", typeof(version))
version = "1.9.0"  # Теперь это строка
println("Новый тип 'version': ", typeof(version))

# Соглашения об именовании: используйте snake_case для переменных и функций
user_count = 100
total_amount = 123.45

# Константы объявляются через const и должны быть ВЕРХНЕМ РЕГИСТРЕ
const PI = 3.141592653589793
const MAX_USERS = 1000

println("\nКонстанты:")
println("PI = ", PI)
println("MAX_USERS = ", MAX_USERS)

# ================ 2. ОСНОВНЫЕ ТИПЫ ДАННЫХ ================
println("\n# ================ 2. ОСНОВНЫЕ ТИПЫ ДАННЫХ ================")
# Целые числа бывают разных размеров
int8_var::Int8 = 127      # 8-битное знаковое целое (-128 до 127)
int16_var::Int16 = 32767  # 16-битное знаковое целое
int32_var::Int32 = 2147483647  # 32-битное знаковое целое
int64_var::Int64 = 9223372036854775807  # 64-битное знаковое целое

println("Целочисленные типы: ")
println("Int8:  ", int8_var,  " (тип:  ", typeof(int8_var),  ") ")
println("Int16:  ", int16_var,  " (тип:  ", typeof(int16_var),  ") ")
println("Int32:  ", int32_var,  " (тип:  ", typeof(int32_var),  ") ")
println("Int64:  ", int64_var,  " (тип:  ", typeof(int64_var),  ") ")

# Беззнаковые целые числа
uint8_var::UInt8 = 0xff  # Шестнадцатеричная нотация
uint32_var::UInt32 = 0xffffffff

println("\nБеззнаковые целые числа:")
println("UInt8: ", uint8_var, " (hex: 0x", string(uint8_var, base=16), ")")
println("UInt32: ", uint32_var, " (hex: 0x", string(uint32_var, base=16), ")")

# Числа с плавающей точкой
float32_var::Float32 = 3.14159f0
float64_var::Float64 = 3.141592653589793

println("\nТипы с плавающей точкой:")
println("Float32: ", float32_var, " (тип: ", typeof(float32_var), ")")
println("Float64: ", float64_var, " (тип: ", typeof(float64_var), ")")

# Научная нотация
scientific_num = 1.23e-4
println("Научная нотация: ", scientific_num)

# Комплексные числа
complex_num = 3 + 4im
println("\nКомплексное число: ", complex_num)
println("Вещественная часть: ", real(complex_num))
println("Мнимая часть: ", imag(complex_num))
println("Модуль: ", abs(complex_num))
println("Сопряженное: ", conj(complex_num))

# Рациональные числа
rational_num = 22//7
println("\nРациональное число: ", rational_num)
println("В виде float: ", Float64(rational_num))

# Булевый тип
bool_true = true
bool_false = false
println("\nБулевы значения: ", bool_true, " и ", bool_false)

# Символы и строки
char_a = 'A'
string_hello = "Привет, Julia!"
println("\nСимвол: ", char_a, " (тип: ", typeof(char_a), ")")
println("Строка: ", string_hello, " (тип: ", typeof(string_hello), ")")

# ================ 3. СТРОКИ И ОПЕРАЦИИ СО СТРОКАМИ ================
println("\n# ================ 3. СТРОКИ И ОПЕРАЦИИ СО СТРОКАМИ ================")
# Конкатенация строк
first_name = "John"
last_name = "Doe"
full_name = first_name * " " * last_name
println("Полное имя: ", full_name)

# Интерполяция строк (предпочтительный метод)
age = 30
info = "Имя: $full_name, Возраст: $age"
println("Интерполированная строка: ", info)

# Многострочные строки
multiline_string = """
Это многострочная строка.
Она может занимать несколько строк.
Очень полезно для форматированного текста.
"""
println("Многострочная строка:\n", multiline_string)

# Операции со строками
text =  "Язык программирования Julia "
println("Исходный текст:  ", text)
println("Длина:  ", length(text))
println("Верхний регистр:  ", uppercase(text))
println("Нижний регистр:  ", lowercase(text))
println("Заглавные буквы:  ", titlecase(text))
println("Содержит 'программирования'?  ", occursin("программирования", text))
println("Заменить 'Julia' на 'System':  ", replace(text, "Julia" => "System"))

# Доступ к символам и подстрокам
println("Первый символ: ", text[1])
println("Последний символ: ", text[end])
println("Подстрока (1:5): ", text[1:5])
println("Последние 8 символов: ", text[(end-8):end])

# ================ 4. ОПЕРАЦИИ ВВОДА/ВЫВОДА ================
println("\n# ================ 4. ОПЕРАЦИИ ВВОДА/ВЫВОДА ================")
# Базовый вывод с println (с новой строкой) и print (без новой строки)
print("Это напечатано без новой строки")
print(" поэтому это продолжается на той же строке\n")

# Форматированный вывод
using Printf  # Необходимо для @sprintf
value = 42
formatted_output = @sprintf("Ответ: %d", value)
println(formatted_output)

# Чтение из стандартного ввода (закомментировано, чтобы избежать блокировки выполнения)
# println("Введите ваше имя: ")
# user_input = readline()
# println("Здравствуйте, $user_input!")

# Чтение чисел из ввода
# println("Введите число: ")
# number_input = parse(Float64, readline())
# println("Вы ввели: $number_input, квадрат: $(number_input^2)")

# Операции с файлами - запись
println("\nОперации с файлами:")
filename = "sample_ru.txt"
open(filename, "w") do file
    write(file, "Привет от Julia!\n")
    write(file, "Это тестовый файл.\n")
    write(file, "Создан для учебных целей.\n")
end
println("Записано содержимое в $filename")

# Операции с файлами - чтение
content = open(filename, "r") do file
    read(file, String)
end
println("Содержимое, прочитанное из $filename:")
println(content)

# Проверка существования файла
println("Существует ли '$filename'? ", isfile(filename))

# Очистка - удаление тестового файла
rm(filename)
println("Удалено $filename")

# ================ 5. УПРАВЛЯЮЩИЕ КОНСТРУКЦИИ ================
println("\n# ================ 5. УПРАВЛЯЮЩИЕ КОНСТРУКЦИИ ================")
# Условные операторы
temperature = 25
if temperature > 30
    println("На улице жарко!")
elseif temperature > 20
    println("На улице тепло! Температура: $temperature °C")
else
    println("На улице прохладно! Температура: $temperature °C")
end

# Тернарный оператор
weather = temperature > 20 ? "тепло" : "прохладно"
println("Погодное условие: $weather")

# Вычисление с коротким замыканием
is_sunny = true
is_warm = temperature > 20
should_go_outside = is_sunny && is_warm  # Вычисляет второе условие только если первое истинно
println("Выйти на улицу? $should_go_outside")

# Циклы - циклы for
println("\nПример цикла for:")
for i in 1:5
    println("Итерация $i")
end

# Цикл for с массивами
fruits = ["яблоко", "банан", "апельсин"]
println("\nСписок фруктов:")
for (index, fruit) in enumerate(fruits)
    println("$index. $fruit")
end

# Цикл while
println("\nПример цикла while:")
counter = 1
while counter <= 3
    println("Счетчик: $counter")
    global counter  # Declare counter as global to modify it inside the loop
    counter += 1
end

# Управление циклом - break и continue
println("\nЦикл с break и continue:")
for i in 1:10
    if i == 3
        continue  # Пропустить остаток этой итерации
    end
    if i == 7
        break  # Выйти из цикла
    end
    println("Число: $i")
end

# Обработка исключений
println("\nПример обработки исключений:")
try
    result = 10 / 0  # Это вызовет ошибку
catch e
    println("Произошла ошибка: ", e)
finally
    println("Это выполняется независимо от того, произошло ли исключение")
end

# Пример безопасного деления
function safe_divide(a, b)
    try
        return a / b
    catch e
        println("Ошибка при делении: ", e)
        return nothing
    end
end

println("Безопасное деление 10/2: ", safe_divide(10, 2))
println("Безопасное деление 10/0: ", safe_divide(10, 0))

# ================ 6. ФУНКЦИИ ================
println("\n# ================ 6. ФУНКЦИИ ================")
# Базовое определение функции
function greet(name)
    return "Здравствуйте, $name !"
end
println(greet("Алиса"))

# Компактный синтаксис функции
square(x) = x^2
println("Квадрат 5: ", square(5))

# Функции с несколькими аргументами
function calculate_area(length, width)
    return length * width
end
area = calculate_area(10, 5)
println("Площадь прямоугольника (10x5): $area")

# Функции с аргументами по умолчанию
function introduce(name, age=25, city="Неизвестно")
    return "Привет, я $name, мне $age лет, из $city."
end
println(introduce("Боб"))
println(introduce("Чарли", 30))
println(introduce("Диана", 28, "Нью-Йорк"))

# Функции с именованными аргументами
function create_profile(; name, age, email="unknown@email.com", phone=nothing)
    profile = Dict(
        "name" => name,
        "age" => age,
        "email" => email,
        "phone" => phone
    )
    return profile
end
profile = create_profile(name="Ева", age=35, email="eve@example.com")
println("Профиль: ", profile)

# Анонимные функции (лямбда-функции)
multiply_by_two = x -> x * 2
println("Результат анонимной функции: ", multiply_by_two(7))

# Пример функций высшего порядка
numbers = [1, 2, 3, 4, 5]
squared_numbers = map(square, numbers)  # Применить функцию square к каждому элементу
println("Исходные числа: ", numbers)
println("Возведенные в квадрат числа: ", squared_numbers)

# Функция фильтрации
even_numbers = filter(x -> x % 2 == 0, numbers)
println("Четные числа: ", even_numbers)

# Функция свертки
sum_of_numbers = reduce(+, numbers)
println("Сумма чисел: ", sum_of_numbers)

# Множественные возвращаемые значения
function divide_with_remainder(dividend, divisor)
    quotient = dividend ÷ divisor  # Целочисленное деление
    remainder = dividend % divisor  # Остаток от деления
    return quotient, remainder
end
quot, rem = divide_with_remainder(17, 5)
println("17 ÷ 5 = $quot с остатком $rem")

println("\n=== Конец Основ Julia Часть 1 ===")
println("В следующих частях мы рассмотрим коллекции, расширенные типы, построение графиков, внешние библиотеки и создание модулей.")