# Основы Julia Часть 2: Коллекции и Расширенные Типы
# Эта программа знакомит пользователей с типами коллекций Julia и расширенной системой типов
# Темы: Массивы, Кортежи, Словари, Множества, Система Типов и Структуры

println("=== Основы Julia Часть 2: Коллекции и Расширенные Типы ===")
println()

# ================ 1. МАССИВЫ ================
println("# ================ 1. МАССИВЫ ================")
# Массивы являются фундаментальными структурами данных в Julia
# Массивы Julia используют 1-индексацию (первый элемент имеет индекс 1)

# Создание массивов
int_array = [1, 2, 3, 4, 5]
float_array = [1.0, 2.0, 3.0, 4.0, 5.0]
mixed_array = [1, 2.0, "three", 4]  # Будет преобразован в тип Any

println("Целочисленный массив: ", int_array)
println("Массив с плавающей точкой: ", float_array)
println("Смешанный массив: ", mixed_array)
println("Тип смешанного массива: ", typeof(mixed_array))

# Массив с конкретным типом
typed_array::Array{Int64, 1} = [10, 20, 30, 40, 50]
println("\nТипизированный массив: ", typed_array)
println("Тип элемента: ", eltype(typed_array))
println("Количество измерений: ", ndims(typed_array))
println("Длина: ", length(typed_array))

# Многомерные массивы
matrix_2x3 = [1 2 3; 4 5 6]  # 2 строки, 3 столбца
println("\nМатрица 2x3:")
println(matrix_2x3)
println("Размер: ", size(matrix_2x3))

matrix_3x3 = [1 2 3; 4 5 6; 7 8 9]
println("\nМатрица 3x3:")
println(matrix_3x3)

# Генерация массивов (comprehension)
squares = [x^2 for x in 1:10]
println("\nКвадраты (1-10) через генератор: ", squares)

# Генерация массивов с условием
even_squares = [x^2 for x in 1:10 if x % 2 == 0]
println("Квадраты четных (1-10): ", even_squares)

# Создание массивов функциями
zeros_array = zeros(5)  # Массив из 5 нулей
ones_array = ones(3, 3)  # Массив 3x3 из единиц
range_array = collect(1:2:10)  # Диапазон от 1 до 10 с шагом 2

println("\nМассив нулей: ", zeros_array)
println("Массив единиц:\n", ones_array)
println("Массив диапазона: ", range_array)

# Доступ к элементам массива
println("\nДоступ к элементам массива:")
println("Первый элемент: ", int_array[1])
println("Последний элемент: ", int_array[end])
println("Элементы 2-4: ", int_array[2:4])
println("Каждый второй элемент: ", int_array[1:2:end])

# Изменение массивов
int_array[1] = 100  # Изменить первый элемент
println("\nПосле изменения первого элемента: ", int_array)

push!(int_array, 6)  # Добавить элемент в конец
println("После push!: ", int_array)

pop!(int_array)  # Удалить последний элемент
println("После pop!: ", int_array)

insert!(int_array, 1, 0)  # Вставить на позицию 1
println("После insert!: ", int_array)

deleteat!(int_array, 1)  # Удалить на позиции 1
println("После deleteat!: ", int_array)

# Операции с массивами
array1 = [1, 2, 3]
array2 = [4, 5, 6]
concatenated = vcat(array1, array2)  # Вертикальная конкатенация
println("\nКонкатенированные массивы: ", concatenated)

# Операции с матрицами
matrix_a = [1 2; 3 4]
matrix_b = [5 6; 7 8]
matrix_sum = matrix_a + matrix_b
matrix_product = matrix_a * matrix_b
println("\nМатрица A:\n", matrix_a)
println("Матрица B:\n", matrix_b)
println("A + B:\n", matrix_sum)
println("A * B:\n", matrix_product)

# ================ 2. КОРТЕЖИ ================
println("\n# ================ 2. КОРТЕЖИ ================")
# Кортежи - это неизменяемые упорядоченные коллекции
# Создаются с помощью круглых скобок или значений, разделенных запятыми

# Создание кортежей
tuple1 = (1, 2, 3)
tuple2 = "a", "b", "c"  # Скобки необязательны
single_element_tuple = (42,)  # Обратите внимание на запятую для одного элемента

println("Кортеж 1: ", tuple1)
println("Кортеж 2: ", tuple2)
println("Кортеж с одним элементом: ", single_element_tuple)
println("Тип: ", typeof(tuple1))

# Доступ к элементам кортежа
println("\nДоступ к элементам кортежа:")
println("Первый элемент: ", tuple1[1])
println("Последний элемент: ", tuple1[end])

# Распаковка кортежа
x, y, z = tuple1
println("\nРаспакованные значения: x=$x, y=$y, z=$z")

# Именованные кортежи
person = (name="Алиса", age=30, city="Москва")
println("\nИменованный кортеж: ", person)
println("Имя: ", person.name)
println("Возраст: ", person.age)

# Кортежи неизменяемы (нельзя модифицировать)
# tuple1[1] = 100  # Это вызовет ошибку!

# ================ 3. СЛОВАРИ ================
println("\n# ================ 3. СЛОВАРИ ================")
# Словари - это пары ключ-значение
# Создаются с помощью Dict() или синтаксиса словарного литерала

# Создание словарей
dict1 = Dict("name" => "Боб", "age" => 25, "city" => "Лондон")
dict2 = Dict(
    "apple" => 1.50,
    "banana" => 0.75,
    "orange" => 2.00
)

println("Словарь 1: ", dict1)
println("Словарь 2: ", dict2)

# Доступ к значениям словаря
println("\nДоступ к значениям словаря:")
println("Имя: ", dict1["name"])
println("Возраст: ", dict1["age"])

# Безопасный доступ с get() (возвращает значение по умолчанию, если ключ не существует)
country = get(dict1, "country", "Неизвестно")
println("Страна (со значением по умолчанию): ", country)

# Добавление и изменение записей
dict1["email"] = "bob@example.com"
println("\nПосле добавления email: ", dict1)

dict1["age"] = 26
println("После обновления возраста: ", dict1)

# Удаление записей
delete!(dict1, "city")
println("После удаления city: ", dict1)

# Операции со словарем
println("\nОперации со словарем:")
println("Ключи: ", keys(dict1))
println("Значения: ", values(dict1))
println("Есть ключ 'name'? ", haskey(dict1, "name"))
println("Количество пар: ", length(dict1))

# Итерация по словарю
println("\nИтерация по словарю:")
for (key, value) in dict1
    println("$key: $value")
end

# ================ 4. МНОЖЕСТВА ================
println("\n# ================ 4. МНОЖЕСТВА ================")
# Множества - это неупорядоченные коллекции уникальных элементов
# Создаются с помощью Set()

# Создание множеств
set1 = Set([1, 2, 3, 4, 5])
set2 = Set([3, 4, 5, 6, 7])

println("Множество 1: ", set1)
println("Множество 2: ", set2)

# Операции с множествами
union_set = union(set1, set2)  # Все элементы из обоих множеств
intersect_set = intersect(set1, set2)  # Общие элементы
diff_set = setdiff(set1, set2)  # Элементы в set1, но не в set2

println("\nОбъединение: ", union_set)
println("Пересечение: ", intersect_set)
println("Разность (set1 - set2): ", diff_set)

# Принадлежность к множеству
println("\nПринадлежность к множеству:")
println("3 в set1? ", 3 in set1)
println("8 в set1? ", 8 in set1)

# Добавление и удаление элементов
push!(set1, 6)
println("\nПосле добавления 6: ", set1)

pop!(set1)  # Удалить произвольный элемент
println("После pop!: ", set1)

# ================ 5. СИСТЕМА ТИПОВ ================
println("\n# ================ 5. СИСТЕМА ТИПОВ ================")
# Julia имеет богатую иерархию типов
# Все типы являются подтипами Any

# Проверка типов
value1 = 42
value2 = 3.14
value3 = "Привет"

println("Тип 42: ", typeof(value1))
println("Тип 3.14: ", typeof(value2))
println("Тип 'Привет': ", typeof(value3))

# Иерархия типов
println("\nИерархия типов:")
println("Int64 <: Integer? ", Int64 <: Integer)
println("Integer <: Real? ", Integer <: Real)
println("Real <: Number? ", Real <: Number)
println("Number <: Any? ", Number <: Any)

# Проверка типа
println("\nПроверка типа:")
println("42 это Integer? ", isa(value1, Integer))
println("3.14 это Real? ", isa(value2, Real))
println("'Привет' это String? ", isa(value3, String))

# Преобразование типов
println("\nПреобразование типов:")
int_to_float = Float64(42)
float_to_int = Int64(round(3.99) ) # Усекает, не округляет
string_to_int = parse(Int64, "123")

println("Int в Float: ", int_to_float, " (тип: ", typeof(int_to_float), ")")
println("Float в Int: ", float_to_int, " (тип: ", typeof(float_to_int), ")")
println("String в Int: ", string_to_int, " (тип: ", typeof(string_to_int), ")")

# ================ 6. ПОЛЬЗОВАТЕЛЬСКИЕ ТИПЫ (СТРУКТУРЫ) ================
println("\n# ================ 6. ПОЛЬЗОВАТЕЛЬСКИЕ ТИПЫ (СТРУКТУРЫ) ================")
# Структуры позволяют определять пользовательские типы данных

# Базовое определение структуры
struct Person
    name::String
    age::Int
    email::String
end

# Создание экземпляров структур
person1 = Person("Алиса", 30, "alice@example.com")
person2 = Person("Боб", 25, "bob@example.com")

println("Персона 1: ", person1)
println("Персона 2: ", person2)

# Доступ к полям структуры
println("\nДоступ к полям структуры:")
println("Имя персоны 1: ", person1.name)
println("Возраст персоны 1: ", person1.age)
println("Email персоны 1: ", person1.email)

# Структуры неизменяемы по умолчанию
# person1.age = 31  # Это вызовет ошибку!

# Изменяемая структура (можно модифицировать после создания)
mutable struct MutablePerson
    name::String
    age::Int
    email::String
end

mutable_person = MutablePerson("Чарли", 35, "charlie@example.com")
println("\nИзменяемая персона: ", mutable_person)

mutable_person.age = 36  # Это работает!
println("После обновления возраста: ", mutable_person)

# Структура со значениями по умолчанию (внутренний конструктор)
struct Point
    x::Float64
    y::Float64
    Point(x, y) = new(x, y)
    Point() = new(0.0, 0.0)  # Конструктор по умолчанию
end

point1 = Point()
point2 = Point(3.0, 4.0)

println("\nТочка по умолчанию: ", point1)
println("Точка со значениями: ", point2)

# ================ 7. ПАРАМЕТРИЧЕСКИЕ ТИПЫ ================
println("\n# ================ 7. ПАРАМЕТРИЧЕСКИЕ ТИПЫ ================")
# Параметрические типы позволяют использовать параметры типов

# Параметрическая структура
struct Container{T}
    value::T
end

int_container = Container{Int}(42)
float_container = Container{Float64}(3.14)
string_container = Container{String}("Привет")

println("Контейнер Int: ", int_container)
println("Контейнер Float: ", float_container)
println("Контейнер String: ", string_container)

# Параметр типа может быть выведен автоматически
inferred_container = Container("Мир")
println("Выведенный контейнер: ", inferred_container)
println("Тип: ", typeof(inferred_container))

# ================ 8. ТИПЫ ОБЪЕДИНЕНИЯ ================
println("\n# ================ 8. ТИПЫ ОБЪЕДИНЕНИЯ ================")
# Типы объединения позволяют переменной быть одного из нескольких типов

# Создание типа объединения
StringOrInt = Union{String, Int}

function process_value(value::StringOrInt)
    if isa(value, String)
        return "Строка: $value"
    else
        return "Целое число: $value"
    end
end

println(process_value("Привет"))
println(process_value(42))

# Тип Nothing (аналог null в других языках)
maybe_value::Union{String, Nothing} = nothing
println("\nВозможное значение: ", maybe_value)

maybe_value = "Теперь имеет значение"
println("Возможное значение после присваивания: ", maybe_value)

# ================ 9. ПСЕВДОНИМЫ ТИПОВ ================
println("\n# ================ 9. ПСЕВДОНИМЫ ТИПОВ ================")
# Псевдонимы типов упрощают чтение сложных типов

const Point2D = Tuple{Float64, Float64}
const StringDict = Dict{String, Any}

point_2d::Point2D = (1.0, 2.0)
string_dict::StringDict = Dict("key1" => "value1", "key2" => 42)

println("Point2D: ", point_2d)
println("StringDict: ", string_dict)

# ================ 10. АБСТРАКТНЫЕ ТИПЫ ================
println("\n# ================ 10. АБСТРАКТНЫЕ ТИПЫ ================")
# Абстрактные типы не могут быть созданы, но могут иметь подтипы

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

println("Круг: ", circle)
println("Прямоугольник: ", rectangle)
println("Сфера: ", sphere)

# Проверка типа с абстрактными типами
println("\nПроверка типа:")
println("Circle это Shape? ", circle isa Shape)
println("Circle это TwoDShape? ", circle isa TwoDShape)
println("Sphere это TwoDShape? ", sphere isa TwoDShape)

# Функция, работающая с любым Shape
function describe_shape(shape::Shape)
    return "Это $(typeof(shape))"
end

println("\nОписание круга: ", describe_shape(circle))
println("Описание прямоугольника: ", describe_shape(rectangle))
println("Описание сферы: ", describe_shape(sphere))

# ================ 11. ВЛОЖЕННЫЕ КОЛЛЕКЦИИ ================
println("\n# ================ 11. ВЛОЖЕННЫЕ КОЛЛЕКЦИИ ================")
# Коллекции могут содержать другие коллекции

# Массив массивов
array_of_arrays = [[1, 2], [3, 4], [5, 6]]
println("Массив массивов: ", array_of_arrays)
println("Первый внутренний массив: ", array_of_arrays[1])
println("Первый элемент первого внутреннего массива: ", array_of_arrays[1][1])

# Словарь со значениями-массивами
inventory = Dict(
    "fruits" => ["яблоко", "банан", "апельсин"],
    "vegetables" => ["морковь", "брокколи", "шпинат"],
    "grains" => ["рис", "пшеница", "овес"]
)

println("\nИнвентарь: ", inventory)
println("Фрукты: ", inventory["fruits"])

# Массив словарей
users = [
    Dict("name" => "Алиса", "age" => 30),
    Dict("name" => "Боб", "age" => 25),
    Dict("name" => "Чарли", "age" => 35)
]

println("\nПользователи:")
for (i, user) in enumerate(users)
    println("Пользователь $i: $(user["name"]), Возраст: $(user["age"])")
end

# ================ 12. СРЕЗЫ И ИНДЕКСАЦИЯ ================
println("\n# ================ 12. СРЕЗЫ И ИНДЕКСАЦИЯ ================")
# Продвинутые срезы и индексация массивов

data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

println("Исходные данные: ", data)
println("Первые 3 элемента: ", data[1:3])
println("Последние 3 элемента: ", data[end-2:end])
println("Каждый 2-й элемент: ", data[1:2:end])
println("В обратном порядке: ", data[end:-1:1])

# Булева индексация
mask = data .> 50  # Создать булеву маску
println("\nЭлементы > 50: ", data[mask])

# Поиск индексов
indices = findall(x -> x > 50, data)
println("Индексы элементов > 50: ", indices)

# ================ 13. ТРАНСЛИЦИЯ (BROADCASTING) ================
println("\n# ================ 13. ТРАНСЛИЦИЯ (BROADCASTING) ================")
# Трансляция применяет операции поэлементно

array_a = [1, 2, 3, 4, 5]
array_b = [10, 20, 30, 40, 50]

# Поэлементные операции (обратите внимание на точку)
sum_array = array_a .+ array_b
product_array = array_a .* array_b
squared_array = array_a .^ 2

println("Массив A: ", array_a)
println("Массив B: ", array_b)
println("A .+ B: ", sum_array)
println("A .* B: ", product_array)
println("A .^ 2: ", squared_array)

# Трансляция с функциями
sqrt_array = sqrt.(array_a)
println("sqrt.(A): ", sqrt_array)

# ================ 14. СОВЕТЫ ПО ПАМЯТИ И ПРОИЗВОДИТЕЛЬНОСТИ ================
println("\n# ================ 14. СОВЕТЫ ПО ПАМЯТИ И ПРОИЗВОДИТЕЛЬНОСТИ ================")
# Предварительно выделяйте память для массивов, когда это возможно
println("Пример предварительного выделения:")
n = 1000000

# Плохо: рост массива в цикле (медленно)
# result_bad = []
# for i in 1:n
#     push!(result_bad, i^2)
# end

# Хорошо: предварительное выделение массива (быстро)
result_good = zeros(Int64, n)
for i in 1:n
    result_good[i] = i^2
end

println("Длина предварительно выделенного массива: ", length(result_good))
println("Первые 10 элементов: ", result_good[1:10])

# Используйте представления вместо копий, когда это возможно
large_array = collect(1:100)
view_array = @view large_array[1:10]  # Создает представление, не копию
copy_array = large_array[1:10]  # Создает копию

println("\nТип представления: ", typeof(view_array))
println("Тип копии: ", typeof(copy_array))

println("\n=== Конец Основ Julia Часть 2 ===")
println("В следующих частях мы рассмотрим построение графиков, визуализацию, внешние библиотеки и создание модулей.")