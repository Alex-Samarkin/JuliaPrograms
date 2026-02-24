# Основы Julia Часть 3: Построение графиков, Визуализация и Работа с Векторами
# Эта программа знакомит пользователей с построением графиков, визуализацией 
# и продвинутыми операциями с векторами в Julia
# Темы: Пакет Plots, операции с векторами, трансляция, мутирующие функции и макросы

println("=== Основы Julia Часть 3: Построение графиков, Визуализация и Работа с Векторами ===")
println()

# Загрузка необходимых пакетов
using Plots
using Statistics
using LinearAlgebra
using Random
using StatsPlots  # Для расширенных статистических графиков

# Установка темы для лучшей визуализации
gr()  # GR бэкенд для быстрого построения графиков

# ============================================================================
# РАЗДЕЛ 1: ПОНИМАНИЕ СПЕЦИАЛЬНОГО СИНТАКСИСА JULIA
# ============================================================================
println("# ================ 1. СПЕЦИАЛЬНЫЙ СИНТАКСИС JULIA ================")

# ----------------------------------------------------------------------------
# 1.1 СИМВОЛ ВОСКЛИЦАНИЯ (!) - МУТИРУЮЩИЕ ФУНКЦИИ
# ----------------------------------------------------------------------------
println("\n# --- 1.1 СИМВОЛ ВОСКЛИЦАНИЯ (!) ---")
#
# В Julia функции, которые модифицируют свои аргументы (мутирующие функции),
# по соглашению заканчиваются символом восклицания (!).
# Это соглашение об именовании, не enforced языком.
#
# Не-мутирующие функции возвращают новый объект без изменения оригинала.
# Мутирующие функции (с !) модифицируют оригинальный объект на месте.

# Пример с массивами
original_array = [1, 2, 3, 4, 5]
println("Исходный массив: ", original_array)

# Не-мутирующая: sort() возвращает новый отсортированный массив
sorted_array = sort(original_array)
println("После sort() - Исходный: ", original_array)
println("После sort() - Новый массив: ", sorted_array)

# Мутирующая: sort!() модифицирует исходный массив
sort!(original_array)
println("После sort!() - Исходный изменён: ", original_array)

# Больше примеров мутирующих vs не-мутирующих функций
array1 = [5, 2, 8, 1, 9]
array2 = copy(array1)

# reverse() vs reverse!()
reversed = reverse(array1)
println("\nreverse() - Исходный: ", array1)
println("reverse() - Результат: ", reversed)

reverse!(array2)
println("reverse!() - Изменён исходный: ", array2)

# append!() vs vcat()
arr1 = [1, 2, 3]
arr2 = [4, 5, 6]
arr3 = copy(arr1)

vcat_result = vcat(arr1, arr2)  # Не-мутирующая
println("\nvcat() - arr1 без изменений: ", arr1)
println("vcat() - Результат: ", vcat_result)

append!(arr3, arr2)  # Мутирующая
println("append!() - arr3 изменён: ", arr3)

# push!() добавляет элемент в конец массива
numbers = [1, 2, 3]
push!(numbers, 4)
println("\nПосле push!(numbers, 4): ", numbers)

# pop!() удаляет и возвращает последний элемент
popped = pop!(numbers)
println("После pop!(numbers) - Удалено: ", popped, ", Массив: ", numbers)

# ----------------------------------------------------------------------------
# 1.2 ОПЕРАТОР ТОЧКА (.) - ТРАНСЛИЦИЯ (BROADCASTING)
# ----------------------------------------------------------------------------
println("\n# --- 1.2 ОПЕРАТОР ТОЧКА (.) - ТРАНСЛИЦИЯ ---")
#
# Оператор точка (.) включает трансляцию, которая применяет операции 
# поэлементно к массивам без явных циклов.
#
# Это одна из самых мощных возможностей Julia для векторизованных операций.

# Базовая арифметика с трансляцией
array_a = [1, 2, 3, 4, 5]
array_b = [10, 20, 30, 40, 50]

# Без точки: попытается выполнить матричные операции (может выдать ошибку)
# С точкой: поэлементные операции

println("Массив A: ", array_a)
println("Массив B: ", array_b)

# Поэлементное сложение
sum_array = array_a .+ array_b
println("\nA .+ B (поэлементно): ", sum_array)

# Поэлементное умножение
product_array = array_a .* array_b
println("A .* B (поэлементно): ", product_array)

# Поэлементное деление
division_array = array_b ./ array_a
println("B ./ A (поэлементно): ", division_array)

# Поэлементное возведение в степень
squared_array = array_a .^ 2
println("A .^ 2 (поэлементно): ", squared_array)

# Трансляция с функциями
println("\nТрансляция с функциями:")
sqrt_array = sqrt.(array_a)
println("sqrt.(A): ", sqrt_array)

log_array = log.(array_b)
println("log.(B): ", log_array)

# Трансляция с пользовательскими функциями
function add_ten(x)
    return x + 10
end

result = add_ten.(array_a)
println("\nadd_ten.(A): ", result)

# Трансляция с несколькими аргументами
function multiply_add(x, y, z)
    return x * y + z
end

result_multi = multiply_add.(array_a, array_b, 5)
println("multiply_add.(A, B, 5): ", result_multi)

# Трансляция со скалярами
scalar_add = array_a .+ 100
println("\nA .+ 100 (скалярная трансляция): ", scalar_add)

scalar_mult = array_a .* 2
println("A .* 2 (скалярная трансляция): ", scalar_mult)

# Сравнение циклов vs трансляция
println("\n--- Производительность: Цикл vs Трансляция ---")

# Версия с циклом
function loop_square(arr)
    result = similar(arr)
    for i in eachindex(arr)
        result[i] = arr[i]^2
    end
    return result
end

# Версия с трансляцией
function broadcast_square(arr)
    return arr .^ 2
end

test_array = rand(1000)
loop_result = loop_square(test_array)
broadcast_result = broadcast_square(test_array)

println("Результаты совпадают: ", loop_result ≈ broadcast_result)

# ----------------------------------------------------------------------------
# 1.3 СИМВОЛ СОБАКА (@) - МАКРОСЫ
# ----------------------------------------------------------------------------
println("\n# --- 1.3 СИМВОЛ СОБАКА (@) - МАКРОСЫ ---")
#
# Макросы в Julia - это код, который генерирует код. Они вычисляются во время 
# парсинга, до выполнения кода. Макросы начинаются с символа @.
#
# Распространённые макросы:
# - @show: печатает выражение и его значение
# - @time: измеряет время выполнения
# - @printf: форматированный вывод (из пакета Printf)
# - @view: создаёт представление вместо копирования
# - @assert: проверка утверждений
# - @macroexpand: показывает, во что разворачивается макрос

# Макрос @show - печатает выражение и значение
println("\nМакрос @show:")
x = 42
@show x
@show x^2
@show typeof(x)

# Макрос @time - измеряет время выполнения
println("\nМакрос @time:")
@time sleep(0.1)

@time begin
    sum_result = sum(1:1000000)
    println("Сумма от 1 до 1.000.000: ", sum_result)
end

# Макрос @printf - форматированный вывод (требует пакет Printf)
using Printf
println("\nМакрос @printf:")
@printf("Pi до 5 знаков: %.5f\n", π)
@printf("Целое: %d, Вещественное: %.2f, Строка: %s\n", 42, 3.14, "Привет")

# Макрос @view - создаёт представление вместо копии (эффективно по памяти)
println("\nМакрос @view:")
large_array = collect(1:1000000)

# Обычный срез создаёт копию
copy_slice = large_array[1:100]
println("Тип обычного среза: ", typeof(copy_slice))

# @view создаёт представление (ссылка на оригинальные данные)
view_slice = @view large_array[1:100]
println("Тип представления: ", typeof(view_slice))

# Представления не выделяют новую память
println("Представление делит данные с оригиналом: ", view_slice[1] == large_array[1])

# Изменение через представление влияет на оригинал
view_slice[1] = 999
println("Исходный массив[1] после изменения через представление: ", large_array[1])

# Макрос @assert - проверка утверждений
println("\nМакрос @assert:")
value = 100
@assert value > 0 "Значение должно быть положительным"
println("Утверждение прошло: value = ", value)

# @macroexpand - показывает разворачивание макроса
println("\nПример @macroexpand:")
println("@show x разворачивается в: ")
@macroexpand @show x

# ============================================================================
# РАЗДЕЛ 2: ВВЕДЕНИЕ В ПОСТРОЕНИЕ ГРАФИКОВ С PLOTS.JL
# ============================================================================
println("\n# ================ 2. ВВЕДЕНИЕ В ПОСТРОЕНИЕ ГРАФИКОВ ================")

# Plots.jl - это метапакет для построения графиков, предоставляющий 
# унифицированный интерфейс к множеству бэкендов (GR, PyPlot, PlotlyJS и др.)

# ----------------------------------------------------------------------------
# 2.1 БАЗОВЫЕ ЛИНЕЙНЫЕ ГРАФИКИ
# ----------------------------------------------------------------------------
println("\n# --- 2.1 БАЗОВЫЕ ЛИНЕЙНЫЕ ГРАФИКИ ---")

# Простой линейный график
x = range(0, 10, length=100)
y = sin.(x)

plot_basic = plot(x, y, 
                  linewidth=2, 
                  label="sin(x)",
                  xlabel="x",
                  ylabel="y",
                  title="Базовый линейный график",
                  legend=:topright,
                  color=:blue,
                  size=(800, 500))

savefig(plot_basic, "plot_basic_line.png")
println("Сохранено: plot_basic_line.png")

# Несколько линий на одном графике
y2 = cos.(x)
y3 = sin.(x) .* cos.(x)

plot_multi = plot(x, y, label="sin(x)", linewidth=2, color=:blue)
plot!(x, y2, label="cos(x)", linewidth=2, color=:red)
plot!(x, y3, label="sin(x)*cos(x)", linewidth=2, color=:green)

plot!(xlabel="x", 
      ylabel="y", 
      title="Несколько функций",
      legend=:topright,
      size=(800, 500))

savefig(plot_multi, "plot_multiple_lines.png")
println("Сохранено: plot_multiple_lines.png")

# ----------------------------------------------------------------------------
# 2.2 ТОЧЕЧНЫЕ ГРАФИКИ (SCATTER)
# ----------------------------------------------------------------------------
println("\n# --- 2.2 ТОЧЕЧНЫЕ ГРАФИКИ ---")

# Генерация случайных данных
Random.seed!(42)
n_points = 50
x_scatter = rand(n_points) .* 10
y_scatter = rand(n_points) .* 10

plot_scatter = scatter(x_scatter, y_scatter,
                       markersize=8,
                       markeralpha=0.6,
                       label="Случайные точки",
                       xlabel="X",
                       ylabel="Y",
                       title="Точечный график",
                       color=:purple,
                       size=(800, 500))

savefig(plot_scatter, "plot_scatter.png")
println("Сохранено: plot_scatter.png")

# ----------------------------------------------------------------------------
# 2.3 СТОЛБЧАТЫЕ ГРАФИКИ (BAR)
# ----------------------------------------------------------------------------
println("\n# --- 2.3 СТОЛБЧАТЫЕ ГРАФИКИ ---")

categories = ["A", "B", "C", "D", "E"]
values = [23, 45, 56, 78, 32]

plot_bar = bar(categories, values,
               label="Значения",
               xlabel="Категория",
               ylabel="Значение",
               title="Столбчатый график",
               color=:orange,
               size=(800, 500))

savefig(plot_bar, "plot_bar.png")
println("Сохранено: plot_bar.png")

# Группированный столбчатый график
values2 = [30, 35, 40, 45, 50]
plot_grouped_bar = bar(categories, [values values2],
                       label=["Группа 1" "Группа 2"],
                       xlabel="Категория",
                       ylabel="Значение",
                       title="Группированный столбчатый график",
                       bar_position=:dodge,
                       size=(800, 500))

savefig(plot_grouped_bar, "plot_grouped_bar.png")
println("Сохранено: plot_grouped_bar.png")

# ----------------------------------------------------------------------------
# 2.4 ГИСТОГРАММЫ
# ----------------------------------------------------------------------------
println("\n# --- 2.4 ГИСТОГРАММЫ ---")

# Генерация нормально распределённых данных
normal_data = randn(1000) .* 5 .+ 10  # mean=10, std=5

plot_histogram = histogram(normal_data,
                           bins=30,
                           label="Нормальное распределение",
                           xlabel="Значение",
                           ylabel="Частота",
                           title="Гистограмма",
                           color=:teal,
                           alpha=0.7,
                           size=(800, 500))

savefig(plot_histogram, "plot_histogram.png")
println("Сохранено: plot_histogram.png")

# ----------------------------------------------------------------------------
# 2.5 ПОДГРАФИКИ (SUBPLOTS)
# ----------------------------------------------------------------------------
println("\n# --- 2.5 ПОДГРАФИКИ ---")

# Создание нескольких подграфиков
x_sub = range(0, 2π, length=100)

p1 = plot(x_sub, sin.(x_sub), label="sin(x)", title="Синус", color=:blue)
p2 = plot(x_sub, cos.(x_sub), label="cos(x)", title="Косинус", color=:red)
p3 = plot(x_sub, tan.(x_sub), label="tan(x)", title="Тангенс", color=:green, 
          ylim=(-5, 5))
p4 = plot(x_sub, sin.(x_sub) .* cos.(x_sub), label="sin(x)*cos(x)", 
          title="Произведение", color=:purple)

plot_subplots = plot(p1, p2, p3, p4, 
                     layout=(2, 2),
                     size=(1000, 800),
                     xlabel="x",
                     ylabel="y")

savefig(plot_subplots, "plot_subplots.png")
println("Сохранено: plot_subplots.png")

# ============================================================================
# РАЗДЕЛ 3: ПРОДВИНУТАЯ ВИЗУАЛИЗАЦИЯ
# ============================================================================
println("\n# ================ 3. ПРОДВИНУТАЯ ВИЗУАЛИЗАЦИЯ ================")

# ----------------------------------------------------------------------------
# 3.1 КАСТОМИЗАЦИЯ ГРАФИКОВ
# ----------------------------------------------------------------------------
println("\n# --- 3.1 КАСТОМИЗАЦИЯ ГРАФИКОВ ---")

x_custom = range(0, 10, length=200)
y_custom = exp.(-x_custom ./ 5) .* sin.(x_custom)

plot_custom = plot(x_custom, y_custom,
                   linewidth=3,
                   linecolor=:darkblue,
                   linestyle=:solid,
                   label="Затухающая синусоида",
                   marker=:circle,
                   markersize=4,
                   markercolor=:red,
                   markeralpha=0.5,
                   xlabel="Время (с)",
                   ylabel="Амплитуда",
                   title="Кастомизированный график с маркерами",
                   titlefontsize=14,
                   legendfontsize=10,
                   guidefontsize=12,
                   tickfontsize=10,
                   grid=:both,
                   gridalpha=0.3,
                   foreground_color_legend=nothing,
                   size=(800, 500))

# Добавление аннотаций
annotate!(plot_custom, 5, 0.5, Plots.text("Область пика", :green, 10))

# Добавление горизонтальных и вертикальных линий
hline!(plot_custom, [0], color=:black, linestyle=:dash, label="Нулевая линия")
vline!(plot_custom, [π, 2π, 3π], color=:gray, linestyle=:dot, label="Кратные π")

savefig(plot_custom, "plot_customized.png")
println("Сохранено: plot_customized.png")

# ----------------------------------------------------------------------------
# 3.2 3D ГРАФИКИ
# ----------------------------------------------------------------------------
println("\n# --- 3.2 3D ГРАФИКИ ---")

# Генерация 3D данных
x_3d = range(-5, 5, length=50)
y_3d = range(-5, 5, length=50)
z_3d = [sin(sqrt(x^2 + y^2)) for x in x_3d, y in y_3d]

plot_3d_surface = surface(x_3d, y_3d, z_3d,
                          xlabel="X",
                          ylabel="Y",
                          zlabel="Z",
                          title="3D График поверхности",
                          camera=(30, 60),
                          size=(800, 600))

savefig(plot_3d_surface, "plot_3d_surface.png")
println("Сохранено: plot_3d_surface.png")

# 3D линейный график
t_3d = range(0, 4π, length=200)
x_3d_line = cos.(t_3d)
y_3d_line = sin.(t_3d)
z_3d_line = t_3d ./ (4π)

plot_3d_line = plot(x_3d_line, y_3d_line, z_3d_line,
                    xlabel="X",
                    ylabel="Y",
                    zlabel="Z",
                    title="3D Спираль",
                    linewidth=3,
                    color=:rainbow,
                    size=(800, 600))

savefig(plot_3d_line, "plot_3d_helix.png")
println("Сохранено: plot_3d_helix.png")

# ----------------------------------------------------------------------------
# 3.3 КОНТУРНЫЕ ГРАФИКИ
# ----------------------------------------------------------------------------
println("\n# --- 3.3 КОНТУРНЫЕ ГРАФИКИ ---")

x_contour = range(-3, 3, length=100)
y_contour = range(-3, 3, length=100)
z_contour = [sin(x) * cos(y) for x in x_contour, y in y_contour]

plot_contour = contour(x_contour, y_contour, z_contour,
                       xlabel="X",
                       ylabel="Y",
                       title="Контурный график",
                       fill=true,
                       color_palette=:viridis,
                       size=(800, 600))

savefig(plot_contour, "plot_contour.png")
println("Сохранено: plot_contour.png")

# ============================================================================
# РАЗДЕЛ 4: ОПЕРАЦИИ С ВЕКТОРАМИ И МАТРИЦАМИ
# ============================================================================
println("\n# ================ 4. ОПЕРАЦИИ С ВЕКТОРАМИ И МАТРИЦАМИ ================")

# ----------------------------------------------------------------------------
# 4.1 СОЗДАНИЕ ВЕКТОРОВ И МАТРИЦ
# ----------------------------------------------------------------------------
println("\n# --- 4.1 СОЗДАНИЕ ВЕКТОРОВ И МАТРИЦ ---")

# Колонные векторы (1D массивы)
vector1 = [1, 2, 3, 4, 5]
vector2 = collect(1:5)
vector3 = collect(range(1, 5, 5)) |> collect

println("Вектор 1: ", vector1)
println("Вектор 2: ", vector2)
println("Вектор 3: ", vector3)

# Строковые векторы (используя транспонирование)
row_vector = vector1'
println("\nСтроковый вектор (транспонирование): ", row_vector)
println("Тип: ", typeof(row_vector))

# Матрицы
matrix1 = [1 2 3; 4 5 6; 7 8 9]
matrix2 = reshape(1:9, 3, 3)

println("\nМатрица 1:")
println(matrix1)
println("\nМатрица 2:")
println(matrix2)

# Специальные матрицы
identity_matrix = I(3)  # Единичная матрица
zeros_matrix = zeros(3, 3)
ones_matrix = ones(3, 3)
random_matrix = rand(3, 3)

println("\nЕдиничная матрица:")
println(identity_matrix)
println("\nМатрица нулей:")
println(zeros_matrix)

# ----------------------------------------------------------------------------
# 4.2 ОПЕРАЦИИ С МАТРИЦАМИ
# ----------------------------------------------------------------------------
println("\n# --- 4.2 ОПЕРАЦИИ С МАТРИЦАМИ ---")

A = [1 2; 3 4]
B = [5 6; 7 8]

println("Матрица A:")
println(A)
println("\nМатрица B:")
println(B)

# Сложение матриц
C = A + B
println("\nA + B:")
println(C)

# Умножение матриц
D = A * B
println("\nA * B (умножение матриц):")
println(D)

# Поэлементное умножение
E = A .* B
println("\nA .* B (поэлементно):")
println(E)

# Транспонирование матрицы
println("\nA' (транспонирование):")
println(A')

# Обратная матрица
A_inv = inv(A)
println("\nA^(-1) (обратная):")
println(A_inv)

# Проверка обратной
identity_check = A * A_inv
println("\nA * A^(-1) (должна быть единичная):")
println(identity_check)

# Определитель
det_A = det(A)
println("\ndet(A) = ", det_A)

# Собственные значения и векторы
eigen_decomp = eigen(A)
println("\nСобственные значения: ", eigen_decomp.values)
println("Собственные векторы:")
println(eigen_decomp.vectors)

# ----------------------------------------------------------------------------
# 4.3 ОПЕРАЦИИ ЛИНЕЙНОЙ АЛГЕБРЫ
# ----------------------------------------------------------------------------
println("\n# --- 4.3 ОПЕРАЦИИ ЛИНЕЙНОЙ АЛГЕБРЫ ---")

# Скалярное произведение
v1 = [1, 2, 3]
v2 = [4, 5, 6]
dot_product = dot(v1, v2)
println("Скалярное произведение ", v1, " и ", v2, " = ", dot_product)

# Векторное произведение (только для 3D векторов)
cross_product = cross(v1, v2)
println("Векторное произведение: ", cross_product)

# Норма (величина)
norm_v1 = norm(v1)
println("Норма v1: ", norm_v1)

# Нормализация вектора
normalized_v1 = v1 / norm(v1)
println("Нормализованный v1: ", normalized_v1)
println("Норма нормализованного: ", norm(normalized_v1))

# Решение линейных систем: Ax = b
A_sys = [2 1; 1 3]
b_sys = [5, 6]
x_solution = A_sys \ b_sys  # Оператор обратной косой черты решает линейные системы
println("\nРешение Ax = b:")
println("A = ", A_sys)
println("b = ", b_sys)
println("x = ", x_solution)

# Проверка решения
verification = A_sys * x_solution
println("Проверка (A*x): ", verification)

# ----------------------------------------------------------------------------
# 4.4 ГЕНЕРАТОРЫ МАССИВОВ И КОМПРЕХЕНШНЫ
# ----------------------------------------------------------------------------
println("\n# --- 4.4 ГЕНЕРАТОРЫ МАССИВОВ ---")

# Базовый компрехеншн
squares = [x^2 for x in 1:10]
println("Квадраты 1-10: ", squares)

# Компрехеншн с условием
even_squares = [x^2 for x in 1:10 if x % 2 == 0]
println("Квадраты чётных 1-10: ", even_squares)

# Многомерный компрехеншн
multiplication_table = [i * j for i in 1:5, j in 1:5]
println("\nТаблица умножения (5x5):")
println(multiplication_table)

# Выражение-генератор (ленивое вычисление)
generator = (x^2 for x in 1:10)
println("\nТип генератора: ", typeof(generator))
println("Первые 5 значений из генератора: ", collect(generator)[1:5])

# ============================================================================
# РАЗДЕЛ 5: ПРАКТИЧЕСКИЕ ПРИМЕРЫ
# ============================================================================
println("\n# ================ 5. ПРАКТИЧЕСКИЕ ПРИМЕРЫ ================")

# ----------------------------------------------------------------------------
# 5.1 ВИЗУАЛИЗАЦИЯ АНАЛИЗА ДАННЫХ
# ----------------------------------------------------------------------------
println("\n# --- 5.1 ВИЗУАЛИЗАЦИЯ АНАЛИЗА ДАННЫХ ---")

# Генерация синтетических экспериментальных данных
Random.seed!(123)
n_samples = 100
time_data = range(0, 10, length=n_samples)
signal_data = sin.(time_data) .+ 0.3 .* randn(n_samples)  # Сигнал с шумом

# График сырых данных
plot_raw = scatter(time_data, signal_data,
                   markersize=4,
                   markeralpha=0.5,
                   label="Сырые данные",
                   color=:lightblue,
                   xlabel="Время",
                   ylabel="Сигнал",
                   title="Экспериментальные данные с шумом",
                   size=(800, 500))

# Добавление сглаженной линии
smoothed_data = [mean(signal_data[max(1,i-2):min(n_samples,i+2)]) for i in 1:n_samples]
plot!(plot_raw, time_data, smoothed_data,
      linewidth=2,
      label="Сглаженные",
      color=:red)

savefig(plot_raw, "plot_data_analysis.png")
println("Сохранено: plot_data_analysis.png")

# ----------------------------------------------------------------------------
# 5.2 СРАВНЕНИЕ ФУНКЦИЙ
# ----------------------------------------------------------------------------
println("\n# --- 5.2 СРАВНЕНИЕ ФУНКЦИЙ ---")

x_compare = range(0.1, 10, length=200)

# Сравнение разных функций роста
linear = x_compare
quadratic = x_compare .^ 2
logarithmic = log.(x_compare)
exponential = exp.(x_compare ./ 5)

plot_compare = plot(x_compare, linear, label="Линейная (x)", linewidth=2, color=:blue)
plot!(x_compare, quadratic, label="Квадратичная (x²)", linewidth=2, color=:red)
plot!(x_compare, logarithmic, label="Логарифмическая (ln(x))", linewidth=2, color=:green)
plot!(x_compare, exponential, label="Экспоненциальная (e^(x/5))", linewidth=2, color=:purple)

plot!(xlabel="x",
      ylabel="f(x)",
      title="Сравнение функций роста",
      legend=:topleft,
      grid=:both,
      gridalpha=0.3,
      size=(800, 500))

savefig(plot_compare, "plot_function_comparison.png")
println("Сохранено: plot_function_comparison.png")

# ----------------------------------------------------------------------------
# 5.3 СТАТИСТИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ
# ----------------------------------------------------------------------------
println("\n# --- 5.3 СТАТИСТИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ ---")

# Генерация выборочных данных для разных групп
Random.seed!(456)
group_a = randn(100) .* 2 .+ 10
group_b = randn(100) .* 2 .+ 12
group_c = randn(100) .* 2 .+ 11

# Box plot
plot_box = boxplot(["Группа A", "Группа B", "Группа C"],
                   [group_a, group_b, group_c],
                   label="",
                   ylabel="Значение",
                   title="Сравнение Box Plot",
                   color=[:blue :red :green],
                   size=(800, 500))

savefig(plot_box, "plot_boxplot.png")
println("Сохранено: plot_boxplot.png")

# Расчёт и отображение статистики
println("\nСтатистическая сводка:")
println("Группа A - Среднее: ", round(mean(group_a), digits=2), 
        ", Стд: ", round(std(group_a), digits=2))
println("Группа B - Среднее: ", round(mean(group_b), digits=2), 
        ", Стд: ", round(std(group_b), digits=2))
println("Группа C - Среднее: ", round(mean(group_c), digits=2), 
        ", Стд: ", round(std(group_c), digits=2))

# ============================================================================
# РАЗДЕЛ 6: СОВЕТЫ ПО ПРОИЗВОДИТЕЛЬНОСТИ
# ============================================================================
println("\n# ================ 6. СОВЕТЫ ПО ПРОИЗВОДИТЕЛЬНОСТИ ================")

# ----------------------------------------------------------------------------
# 6.1 ПРЕДВАРИТЕЛЬНОЕ ВЫДЕЛЕНИЕ ПАМЯТИ
# ----------------------------------------------------------------------------
println("\n# --- 6.1 ПРЕДВАРИТЕЛЬНОЕ ВЫДЕЛЕНИЕ ПАМЯТИ ---")

n = 10000

# Плохо: рост массива в цикле (медленно)
function bad_growth(n)
    result = []
    for i in 1:n
        push!(result, i^2)
    end
    return result
end

# Хорошо: предварительное выделение массива (быстро)
function good_growth(n)
    result = zeros(Int64, n)
    for i in 1:n
        result[i] = i^2
    end
    return result
end

# Тест обоих
@time bad_result = bad_growth(n)
@time good_result = good_growth(n)

println("Результаты совпадают: ", bad_result == good_result)

# ----------------------------------------------------------------------------
# 6.2 ПРЕДСТАВЛЕНИЯ VS КОПИИ
# ----------------------------------------------------------------------------
println("\n# --- 6.2 ПРЕДСТАВЛЕНИЯ VS КОПИИ ---")

large_data = rand(10000, 100)

# Копия (выделяет новую память)
@time copy_data = large_data[:, 1:50]

# Представление (без выделения)
@time view_data = @view large_data[:, 1:50]

println("Тип копии: ", typeof(copy_data))
println("Тип представления: ", typeof(view_data))

# ----------------------------------------------------------------------------
# 6.3 ТРАНСЛИЦИЯ VS ЦИКЛЫ
# ----------------------------------------------------------------------------
println("\n# --- 6.3 ТРАНСЛИЦИЯ VS ЦИКЛЫ ---")

test_data = rand(10000)

# Версия с циклом
function loop_version(arr)
    result = similar(arr)
    for i in eachindex(arr)
        result[i] = sqrt(arr[i]) + log(arr[i] + 1)
    end
    return result
end

# Версия с трансляцией
function broadcast_version(arr)
    return sqrt.(arr) .+ log.(arr .+ 1)
end

@time loop_result = loop_version(test_data)
@time broadcast_result = broadcast_version(test_data)

println("Результаты совпадают: ", loop_result ≈ broadcast_result)

# ============================================================================
# РАЗДЕЛ 7: СВОДКА И УПРАЖНЕНИЯ
# ============================================================================
println("\n# ================ 7. СВОДКА И УПРАЖНЕНИЯ ================")

println("""
СВОДКА ЧАСТИ 3:

Ключевые концепции, рассмотренные:
1. Соглашение с восклицанием (!) для мутирующих функций
2. Оператор точка (.) для трансляции
3. Символ собака (@) для макросов
4. Базовое и продвинутое построение графиков с Plots.jl
5. Операции с векторами и матрицами
6. Линейная алгебра с LinearAlgebra
7. Советы по оптимизации производительности

Сгенерированные файлы графиков:
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

УПРАЖНЕНИЯ ДЛЯ ПРАКТИКИ:

1. Создайте график, показывающий траекторию снаряда с разными 
   начальными углами (30°, 45°, 60°).

2. Используйте трансляцию для создания таблицы умножения от 1 до 10.

3. Создайте функцию, которая нормализует любой вектор до единичной длины, 
   затем протестируйте её со случайными векторами.

4. Сгенерируйте 1000 случайных точек и создайте 2D гистограмму, 
   показывающую распределение их плотности.

5. Сравните производительность реализации на циклах vs трансляции 
   для умножения матриц.

6. Создайте анимированный график, показывающий движущуюся синусоиду 
   во времени.

7. Используйте @view для создания нескольких представлений большого 
   массива и модифицируйте их, чтобы увидеть, как изменения влияют 
   на оригинал.

8. Создайте панель с 4 подграфиками, показывающими разные 
   статистические распределения (нормальное, равномерное, 
   экспоненциальное, биномиальное).
""")

println("\n=== Конец Основ Julia Часть 3 ===")
println("В Части 4 мы применим эти концепции к биологическому моделированию:")
println("- Динамика инсулин-глюкоза")
println("- Моделирование иммунного ответа")
println("- Эпидемиологические модели (SIR, SEIR)")
println("- Фармакокинетические модели")