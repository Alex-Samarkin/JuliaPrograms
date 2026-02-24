# Основы Julia Часть 4: Организация Проекта, Модули и Пользовательские Типы
# Эта программа охватывает структуру проекта, создание модулей, пользовательские типы,
# и демонстрирует создание переиспользуемого статистического модуля
# Темы: Организация проекта, модули, пакеты, пользовательские типы,
#       описательная статистика, статистическая визуализация

println("=== Основы Julia Часть 4: Организация Проекта, Модули и Пользовательские Типы ===")
println()

# ============================================================================
# РАЗДЕЛ 1: ОРГАНИЗАЦИЯ ПРОЕКТА В JULIA
# ============================================================================
println("# ================ 1. ОРГАНИЗАЦИЯ ПРОЕКТА ================")

# Проекты Julia следуют стандартной структуре:
#
# MyProject/
# ├── Project.toml          # Метаданные проекта и зависимости
# ├── Manifest.toml         # Точные версии всех зависимостей
# ├── src/                  # Исходный код
# │   ├── MyModule.jl       # Файлы модулей
# │   └── ...
# ├── test/                 # Тестовые файлы
# │   ├── runtests.jl
# │   └── ...
# ├── docs/                 # Документация
# │   ├── make.jl
# │   └── src/
# └── README.md             # Описание проекта

println("""
Структура проекта Julia:

MyProject/
├── Project.toml          # Метаданные проекта и зависимости
├── Manifest.toml         # Точные версии всех зависимостей
├── src/                  # Исходный код
│   ├── MyModule.jl       # Файлы модулей
│   └── ...
├── test/                 # Тестовые файлы
│   ├── runtests.jl
│   └── ...
├── docs/                 # Документация
│   ├── make.jl
│   └── src/
└── README.md             # Описание проекта
""")

# Создание проекта с помощью Pkg
println("# Создание проекта:")
println("  В REPL: ] generate MyProject")
println("  Или: using Pkg; Pkg.generate(\"MyProject\")")

# Пример содержимого Project.toml
println("\n# Пример Project.toml:")
println("""
[deps]
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
""")

# ============================================================================
# РАЗДЕЛ 2: МОДУЛИ И ПАКЕТЫ
# ============================================================================
println("\n# ================ 2. МОДУЛИ И ПАКЕТЫ ================")

# ----------------------------------------------------------------------------
# 2.1 ПОНИМАНИЕ МОДУЛЕЙ
# ----------------------------------------------------------------------------
println("\n# --- 2.1 ПОНИМАНИЕ МОДУЛЕЙ ---")
#
# Модули в Julia - это пространства имён для организации кода
# Они помогают избежать конфликтов имён и организуют связанную функциональность

# Базовая структура модуля
module ExampleModule
    # Экспортируемые функции (видны вне модуля)
    export greet, calculate_sum
    
    # Внутренняя функция (не экспортируется)
    function _internal_helper(x)
        return x * 2
    end
    
    # Экспортируемые функции
    greet(name) = "Привет, $name !"
    calculate_sum(a, b) = a + b
end

println("Модуль создан: ExampleModule")
println("Приветствие из модуля: ", ExampleModule.greet("Пользователь"))
println("Сумма: ", ExampleModule.calculate_sum(5, 3))

# ----------------------------------------------------------------------------
# 2.2 USING И IMPORT
# ----------------------------------------------------------------------------
println("\n# --- 2.2 USING И IMPORT ---")

# Различные способы использования модулей:
#
# 1. using ModuleName - приносит все экспортируемые имена в область видимости
# 2. using ModuleName: name1, name2 - приносит конкретные имена
# 3. import ModuleName - приносит модуль в область видимости (использовать ModuleName.name)
# 4. import ModuleName: name1, name2 - импортировать конкретные имена для расширения

# Пример с модулем Statistics
using Statistics

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
println("Данные: ", data)
println("Среднее: ", mean(data))
println("Медиана: ", median(data))
println("Стд: ", std(data))
println("Дисперсия: ", var(data))

# ----------------------------------------------------------------------------
# 2.3 СОЗДАНИЕ ПОЛЬЗОВАТЕЛЬСКИХ МОДУЛЕЙ
# ----------------------------------------------------------------------------
println("\n# --- 2.3 СОЗДАНИЕ ПОЛЬЗОВАТЕЛЬСКИХ МОДУЛЕЙ ---")

# Мы создадим комплексный статистический модуль ниже
# Это демонстрирует лучшие практики создания модулей

# ============================================================================
# РАЗДЕЛ 3: СОЗДАНИЕ СТАТИСТИЧЕСКОГО МОДУЛЯ
# ============================================================================
println("\n# ================ 3. СОЗДАНИЕ СТАТИСТИЧЕСКОГО МОДУЛЯ ================")

# Определяем наш пользовательский модуль StatisticsTools
module StatisticsTools

# Экспортируем публичный API
export 
    # Типы
    StatisticalSummary,
    # Функции
    describe,
    summary_stats,
    outlier_detection,
    normalize,
    standardize,
    # Функции построения графиков
    plot_histogram,
    plot_boxplot,
    plot_qq,
    plot_distribution,
    # Вспомогательные функции
    missing_count,
    unique_count,
    correlation_matrix

# Импорт необходимых пакетов
using Statistics
using StatsBase
using Plots
using Distributions
using LinearAlgebra

# ============================================================================
# ПОЛЬЗОВАТЕЛЬСКИЕ ТИПЫ
# ============================================================================

"""
    StatisticalSummary

Структура, содержащая комплексные описательные статистики для набора данных.

# Поля
- `n::Int`: Количество наблюдений
- `missing_count::Int`: Количество пропущенных значений
- `mean::Float64`: Среднее арифметическое
- `median::Float64`: Медианное значение
- `std::Float64`: Стандартное отклонение
- `var::Float64`: Дисперсия
- `min::Float64`: Минимальное значение
- `max::Float64`: Максимальное значение
- `q25::Float64`: 25-й перцентиль
- `q75::Float64`: 75-й перцентиль
- `skewness::Float64`: Коэффициент асимметрии
- `kurtosis::Float64`: Коэффициент эксцесса
- `cv::Float64`: Коэффициент вариации (std/mean)
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

# Пользовательский метод show для красивого вывода
function Base.show(io::IO, s::StatisticalSummary)
    println(io, "StatisticalSummary:")
    println(io, "  Наблюдений:       $(s.n)")
    println(io, "  Пропущено:        $(s.missing_count)")
    println(io, "  Среднее:          $(round(s.mean, digits=4))")
    println(io, "  Медиана:          $(round(s.median, digits=4))")
    println(io, "  Стд отклонение:   $(round(s.std, digits=4))")
    println(io, "  Дисперсия:        $(round(s.var, digits=4))")
    println(io, "  Мин:              $(round(s.min, digits=4))")
    println(io, "  Макс:             $(round(s.max, digits=4))")
    println(io, "  Q25 (25%):        $(round(s.q25, digits=4))")
    println(io, "  Q75 (75%):        $(round(s.q75, digits=4))")
    println(io, "  Асимметрия:       $(round(s.skewness, digits=4))")
    println(io, "  Эксцесс:          $(round(s.kurtosis, digits=4))")
    println(io, "  Коэф. вариации:   $(round(s.cv, digits=4))")
end

# ============================================================================
# ФУНКЦИИ ОПИСАТЕЛЬНОЙ СТАТИСТИКИ
# ============================================================================

"""
    summary_stats(data::Vector{<:Real})

Вычислить комплексные описательные статистики для вектора.

# Аргументы
- `data`: Входной вектор вещественных чисел

# Возвращает
- `StatisticalSummary`: Структура со всеми статистиками
"""
function summary_stats(data::Vector{<:Real})
    # Обработка пропущенных значений
    clean_data = filter(!isnan, data)
    n_missing = length(data) - length(clean_data)
    n = length(data)
    
    if length(clean_data) == 0
        error("Нет валидных точек данных")
    end
    
    # Базовые статистики
    mean_val = mean(clean_data)
    median_val = median(clean_data)
    std_val = std(clean_data)
    var_val = var(clean_data)
    min_val = minimum(clean_data)
    max_val = maximum(clean_data)
    
    # Квантили
    q25 = quantile(clean_data, 0.25)
    q75 = quantile(clean_data, 0.75)
    
    # Асимметрия и эксцесс (используя StatsBase)
    skew_val = skewness(clean_data)
    kurt_val = kurtosis(clean_data)
    
    # Коэффициент вариации
    cv_val = mean_val != 0 ? std_val / abs(mean_val) : NaN
    
    return StatisticalSummary(
        n, n_missing, mean_val, median_val, std_val, var_val,
        min_val, max_val, q25, q75, skew_val, kurt_val, cv_val
    )
end

"""
    describe(data::Vector{<:Real})

Вывести форматированное описание статистик данных.

# Аргументы
- `data`: Входной вектор
"""
function describe(data::Vector{<:Real})
    stats = summary_stats(data)
    show(stdout, stats)
    return stats
end

"""
    outlier_detection(data::Vector{<:Real}; method::String="iqr")

Обнаружить выбросы в данных.

# Аргументы
- `data`: Входной вектор
- `method`: "iqr" (межквартильный диапазон) или "zscore" (Z-оценка > 3)

# Возвращает
- `Tuple`: (индексы выбросов, значения выбросов, чистые данные)
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
        error("Неизвестный метод: $method. Используйте 'iqr' или 'zscore'")
    end
    
    outlier_indices = findall(outlier_mask)
    outlier_values = clean_data[outlier_mask]
    clean_values = clean_data[.!outlier_mask]
    
    return outlier_indices, outlier_values, clean_values
end

"""
    normalize(data::Vector{<:Real})

Нормализовать данные к диапазону [0, 1] (min-max масштабирование).

# Аргументы
- `data`: Входной вектор

# Возвращает
- `Vector{Float64}`: Нормализованные данные
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

Стандартизировать данные (z-оценка нормализация).

# Аргументы
- `data`: Входной вектор

# Возвращает
- `Vector{Float64}`: Стандартизированные данные (mean=0, std=1)
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

Подсчитать пропущенные значения (NaN, nothing, missing).

# Аргументы
- `data`: Входной вектор

# Возвращает
- `Int`: Количество пропущенных значений
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

Подсчитать уникальные значения в данных.

# Аргументы
- `data`: Входной вектор

# Возвращает
- `Int`: Количество уникальных значений
"""
function unique_count(data::Vector)
    return length(unique(data))
end

"""
    correlation_matrix(data::Matrix{<:Real})

Вычислить матрицу корреляции для нескольких переменных.

# Аргументы
- `data`: Матрица, где столбцы - переменные

# Возвращает
- `Matrix{Float64}`: Матрица корреляции
"""
function correlation_matrix(data::Matrix{<:Real})
    return cor(data)
end

# ============================================================================
# ФУНКЦИИ ПОСТРОЕНИЯ ГРАФИКОВ
# ============================================================================

"""
    plot_histogram(data::Vector{<:Real}; kwargs...)

Создать гистограмму с кривой плотности.

# Аргументы
- `data`: Входной вектор
- `kwargs`: Дополнительные аргументы построения
"""
function plot_histogram(data::Vector{<:Real}; kwargs...)
    clean_data = filter(!isnan, data)
    
    # Создаём график
    p = histogram(clean_data, 
                  normalize=true,
                  alpha=0.6,
                  label="Гистограмма",
                  xlabel="Значение",
                  ylabel="Плотность",
                  title="Гистограмма с кривой плотности",
                  color=:blue,
                  kwargs...)
    
    # Добавить кривую плотности
    density_curve = Normal(mean(clean_data), std(clean_data))
    x_range = range(minimum(clean_data), maximum(clean_data), length=200)
    plot!(p, x_range, pdf.(density_curve, x_range), 
          linewidth=3, 
          label="Нормальная аппроксимация",
          color=:red)
    
    return p
end

"""
    plot_boxplot(data::Vector{<:Real}; label::String="Данные")

Создать box plot.

# Аргументы
- `data`: Входной вектор
- `label`: Метка для графика
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

Создать Q-Q график (квантиль-квантиль) для проверки нормальности.

# Аргументы
- `data`: Входной вектор
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

Создать комплексный график распределения (гистограмма + boxplot + плотность).

# Аргументы
- `data`: Входной вектор
"""
function plot_distribution(data::Vector{<:Real})
    clean_data = filter(!isnan, data)
    
    # Гистограмма
    fig_hist = histogram(clean_data, 
                         normalize=true,
                         alpha=0.6,
                         label="Гистограмма",
                         xlabel="Значение",
                         ylabel="Плотность",
                         title="Гистограмма",
                         color=:blue,
                         legend=:topright)
    
    density_curve = Normal(mean(clean_data), std(clean_data))
    x_range = range(minimum(clean_data), maximum(clean_data), length=200)
    plot!(fig_hist, x_range, pdf.(density_curve, x_range), 
          linewidth=2, 
          label="Нормальная аппроксимация",
          color=:red)
    
    # Box plot
    fig_box = boxplot(["Данные"], [clean_data],
                      label="",
                      ylabel="Значение",
                      title="Box Plot",
                      color=:green)
    
    # Сортированные значения
    fig_sorted = plot(sort(clean_data),
                      label="Сортированные данные",
                      xlabel="Индекс",
                      ylabel="Значение",
                      title="Сортированные значения",
                      color=:purple,
                      linewidth=2)
    
    # Объединить графики
    combined = plot(fig_hist, fig_box, fig_sorted, layout=(1, 3), size=(1200, 400))
    
    return combined
end

end # module StatisticsTools

println("Модуль StatisticsTools успешно создан!")

# ============================================================================
# РАЗДЕЛ 4: ИСПОЛЬЗОВАНИЕ СТАТИСТИЧЕСКОГО МОДУЛЯ
# ============================================================================
println("\n# ================ 4. ИСПОЛЬЗОВАНИЕ СТАТИСТИЧЕСКОГО МОДУЛЯ ================")

# Импорт нашего пользовательского модуля
using .StatisticsTools

# Генерация тестовых данных
using Random
Random.seed!(42)

# Создать разные распределения для тестирования
normal_data = randn(500) .* 10 .+ 50      # Нормальное: mean=50, std=10
skewed_data = randexp(500) .* 5           # Экспоненциальное (скошенное)
uniform_data = rand(500) .* 100           # Равномерное: 0-100

# Добавить выбросы и пропущенные значения
normal_data_with_outliers = vcat(normal_data, [150, 160, -50])
normal_data_with_missing = vcat(normal_data, [NaN, NaN, NaN])

println("\n--- Тестирование с нормальным распределением ---")
println("Длина данных: ", length(normal_data))

# Получить описательные статистики
stats_normal = StatisticsTools.describe(normal_data)

# Создать графики
println("\nГенерация графиков...")

# Гистограмма с плотностью
plot_hist = StatisticsTools.plot_histogram(normal_data)
display(plot_hist)
savefig(plot_hist, "stats_histogram.png")
println("Сохранено: stats_histogram.png")

# Box plot
plot_box = StatisticsTools.plot_boxplot(normal_data, label="Нормальные данные")
display(plot_box)
savefig(plot_box, "stats_boxplot.png")
println("Сохранено: stats_boxplot.png")

# Q-Q plot
plot_qq_plot = StatisticsTools.plot_qq(normal_data)
display(plot_qq_plot)
savefig(plot_qq_plot, "stats_qq.png")
println("Сохранено: stats_qq.png")

# Комплексный график распределения
plot_dist = StatisticsTools.plot_distribution(normal_data)
display(plot_dist)
savefig(plot_dist, "stats_distribution.png")
println("Сохранено: stats_distribution.png")

# Тестирование обнаружения выбросов
println("\n--- Обнаружение выбросов ---")
outlier_indices, outlier_values, clean_values = StatisticsTools.outlier_detection(
    normal_data_with_outliers, 
    method="iqr"
)
println("Найдено выбросов: ", length(outlier_values))
println("Значения выбросов: ", outlier_values)
println("Длина чистых данных: ", length(clean_values))

# Тестирование нормализации
println("\n--- Нормализация ---")
normalized = StatisticsTools.normalize(normal_data)
println("Исходный диапазон: [$(minimum(normal_data)), $(maximum(normal_data))]")
println("Нормализованный диапазон: [$(minimum(normalized)), $(maximum(normalized))]")
println("Нормализованное среднее: ", round(mean(normalized), digits=4))

# Тестирование стандартизации
println("\n--- Стандартизация ---")
standardized = StatisticsTools.standardize(normal_data)
println("Стандартизированное среднее: ", round(mean(standardized), digits=4))
println("Стандартизированное std: ", round(std(standardized), digits=4))

# Тестирование подсчёта пропущенных
println("\n--- Пропущенные значения ---")
println("Количество пропущенных: ", StatisticsTools.missing_count(normal_data_with_missing))

# Тестирование подсчёта уникальных
println("\n--- Уникальные значения ---")
discrete_data = rand(1:10, 100)
println("Уникальных значений в дискретных данных: ", StatisticsTools.unique_count(discrete_data))

# Тестирование матрицы корреляции
println("\n--- Матрица корреляции ---")
correlation_data = randn(100, 3)
correlation_data[:, 2] = correlation_data[:, 1] .* 0.8 .+ randn(100)
correlation_data[:, 3] = correlation_data[:, 1] .* 0.5 .+ randn(100)
corr_matrix = StatisticsTools.correlation_matrix(correlation_data)
println("Матрица корреляции (3 переменные):")
println(round.(corr_matrix, digits=3))

# ============================================================================
# РАЗДЕЛ 5: СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ
# ============================================================================
println("\n# ================ 5. СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ ================")

# Сравнить разные распределения
println("\n--- Сравнение нормального, скошенного и равномерного распределений ---")

stats_skewed = StatisticsTools.summary_stats(skewed_data)
stats_uniform = StatisticsTools.summary_stats(uniform_data)

println("\nНормальное распределение:")
println("  Среднее: $(round(stats_normal.mean, digits=2)), Стд: $(round(stats_normal.std, digits=2))")
println("  Асимметрия: $(round(stats_normal.skewness, digits=3)), Эксцесс: $(round(stats_normal.kurtosis, digits=3))")

println("\nСкошенное (экспоненциальное) распределение:")
println("  Среднее: $(round(stats_skewed.mean, digits=2)), Стд: $(round(stats_skewed.std, digits=2))")
println("  Асимметрия: $(round(stats_skewed.skewness, digits=3)), Эксцесс: $(round(stats_skewed.kurtosis, digits=3))")

println("\nРавномерное распределение:")
println("  Среднее: $(round(stats_uniform.mean, digits=2)), Стд: $(round(stats_uniform.std, digits=2))")
println("  Асимметрия: $(round(stats_uniform.skewness, digits=3)), Эксцесс: $(round(stats_uniform.kurtosis, digits=3))")

# Создать график сравнения
plot_compare = histogram(normal_data, normalize=true, alpha=0.4, label="Нормальное", color=:blue)
histogram!(plot_compare, skewed_data, normalize=true, alpha=0.4, label="Скошенное", color=:red)
histogram!(plot_compare, uniform_data, normalize=true, alpha=0.4, label="Равномерное", color=:green)
plot!(plot_compare, xlabel="Значение", ylabel="Плотность", title="Сравнение распределений", size=(800, 500))
display(plot_compare)
savefig(plot_compare, "stats_distribution_compare.png")
println("\nСохранено: stats_distribution_compare.png")

# ============================================================================
# РАЗДЕЛ 6: РЕКОМЕНДУЕМЫЕ ПАКЕТЫ JULIA
# ============================================================================
println("\n# ================ 6. РЕКОМЕНДУЕМЫЕ ПАКЕТЫ JULIA ================")

println("""
╔══════════════════════════════════════════════════════════════════════════╗
║                    РЕКОМЕНДУЕМЫЕ ПАКЕТЫ JULIA                            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  ХРАНЕНИЕ ДАННЫХ И ТАБЛИЦЫ                                               ║
║  ─────────────────────                                                   ║
║  • DataFrames.jl        - Манипуляция DataFrame (как pandas/R)           ║
║  • CSV.jl               - Чтение/запись CSV файлов                       ║
║  • ExcelFiles.jl        - Чтение/запись Excel файлов                     ║
║  • JSON.jl              - Парсинг и запись JSON                          ║
║  • Arrow.jl             - Формат Apache Arrow для быстрого обмена        ║
║  • SQLite.jl            - Интерфейс к базе данных SQLite                 ║
║  • JDF.jl               - Формат Julia Data Frame (быстрый нативный)     ║
║                                                                          ║
║  МАТЕМАТИЧЕСКОЕ МОДЕЛИРОВАНИЕ И СИМУЛЯЦИЯ                                ║
║  ─────────────────────────────────────                                   ║
║  • DifferentialEquations.jl - Комплексные решатели ОДУ/УЧП/СДУ           ║
║  • ModelingToolkit.jl   - Символьное моделирование и автодифференцирование║
║  • Catalyst.jl          - Моделирование химических реакций               ║
║  • AgentBasedModels.jl  - Фреймворк агентного моделирования              ║
║  • DifferentialEquations.jl - Стохастические и запаздывающие ДУ          ║
║                                                                          ║
║  РЕШЕНИЕ УРАВНЕНИЙ                                                       ║
║  ──────────────────                                                      ║
║  • NLsolve.jl           - Решение нелинейных уравнений                   ║
║  • LinearSolve.jl       - Решатели линейных систем                       ║
║  • Roots.jl             - Алгоритмы поиска корней                        ║
║  • Optim.jl             - Алгоритмы оптимизации                          ║
║  • JuMP.jl              - Математическое оптимизационное программирование║
║                                                                          ║
║  СТАТИСТИКА И ВИЗУАЛИЗАЦИЯ                                               ║
║  ─────────────────────────────                                           ║
║  • Plots.jl             - Метапакет для графиков (унифицированный интерфейс)║
║  • Makie.jl             - Высокопроизводительная интерактивная визуализация║
║  • StatsBase.jl         - Базовые статистические функции                 ║
║  • Distributions.jl     - Вероятностные распределения                    ║
║  • HypothesisTests.jl   - Статистическая проверка гипотез                ║
║  • GLM.jl               - Обобщённые линейные модели                     ║
║  • MixedModels.jl       - Модели смешанных эффектов                      ║
║  • Turing.jl            - Байесовский вывод с MCMC                       ║
║  • RDatasets.jl         - Коллекция датасетов для примеров               ║
║                                                                          ║
║  МАШИННОЕ ОБУЧЕНИЕ                                                       ║
║  ───────────────────                                                     ║
║  • MLJ.jl               - Фреймворк машинного обучения (как scikit-learn)║
║  • Flux.jl              - Библиотека глубокого обучения                  ║
║  • Metalhead.jl         - Предобученные модели глубокого обучения        ║
║  • Clustering.jl        - Алгоритмы кластеризации                        ║
║  • DimensionalReduction.jl - PCA, t-SNE, UMAP                           ║
║                                                                          ║
║  ВРЕМЕННЫЕ РЯДЫ                                                          ║
║  ─────────────────                                                       ║
║  • TimeSeries.jl        - Структуры данных временных рядов               ║
║  • Temporal.jl          - Анализ временных рядов                         ║
║  • ARCHModels.jl        - Модели ARCH/GARCH для волатильности            ║
║  • StateSpaceModels.jl  - Модели пространства состояний и фильтры Калмана║
║                                                                          ║
║  БИОЛОГИЯ И МЕДИЦИНА                                                     ║
║  ─────────────────────────────                                           ║
║  • BioJulia (BioSequences.jl) - Анализ биологических последовательностей ║
║  • Pharmacometrics.jl   - Фармакокинетическое/фармакодинамическое моделирование║
║  • SBML.jl              - Поддержка языка SBML                           ║
║  • CellML.jl            - Поддержка моделей CellML                       ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# Команды установки
println("\n# Команды установки:")
println("  using Pkg")
println("  Pkg.add([\"DataFrames\", \"CSV\", \"Plots\", \"DifferentialEquations\"])")
println("  Pkg.add([\"StatsBase\", \"Distributions\", \"HypothesisTests\"])")
println("  Pkg.add([\"Makie\", \"MLJ\", \"Flux\"])")

# ============================================================================
# РАЗДЕЛ 7: СРАВНЕНИЕ MAKIE И PLOTS
# ============================================================================
println("\n# ================ 7. MAKIE VS PLOTS ================")

println("""
┌─────────────────────────────────────────────────────────────────────────┐
│                    СРАВНЕНИЕ PLOTS.JL И MAKIE.JL                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PLOTS.JL                                                               │
│  ─────────                                                              │
│  ✓ Зрелый и стабильный                                                  │
│  ✓ Простой API, легко изучить                                           │
│  ✓ Множество бэкендов (GR, PyPlot, PlotlyJS и др.)                      │
│  ✓ Хорош для стандартных научных графиков                               │
│  ✓ Низкий порог входа                                                   │
│  ✗ Медленнее для больших датасетов                                      │
│  ✗ Ограниченная интерактивность                                         │
│  ✗ Меньше возможностей для сложной кастомизации                         │
│                                                                         │
│  MAKIE.JL                                                               │
│  ─────────                                                              │
│  ✓ Высокая производительность (GPU-ускорение)                           │
│  ✓ Высоко интерактивный (зум, панорамирование, наведение)               │
│  ✓ Красивое стилевое оформление по умолчанию                            │
│  ✓ Отлично для сложной 3D визуализации                                  │
│  ✓ Анимации в реальном времени                                          │
│  ✓ Современный рендеринг на основе OpenGL                               │
│  ✗ Более высокий порог входа                                            │
│  ✗ Новее пакет (меньше ресурсов сообщества)                             │
│  ✗ Требует больше настройки                                             │
│                                                                         │
│  РЕКОМЕНДАЦИЯ:                                                          │
│  ──────────────                                                         │
│  • Используйте Plots.jl для: Быстрого исследования, стандартных графиков│
│  • Используйте Makie.jl для: Интерактивных дашбордов, сложного 3D,      │
│    анимаций                                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
""")

# Пример кода Makie (закомментировано - требует установки)
println("\n# Пример кода Makie (требует: Pkg.add(\"GLMakie\")):")
println("""
using GLMakie

# Простой интерактивный график
fig = Figure()
ax = Axis(fig[1, 1], xlabel="x", ylabel="y")
lines!(ax, 0..10, sin)
scatter!(ax, 1:10, rand(10))
display(fig)

# 3D поверхность
fig2 = Figure()
ax2 = Axis3(fig2[1, 1])
surface!(ax2, -5:0.1:5, -5:0.1:5, (x,y) -> sin(sqrt(x^2 + y^2)))
display(fig2)
""")

# ============================================================================
# РАЗДЕЛ 8: ЛУЧШИЕ ПРАКТИКИ ДЛЯ ПРОЕКТОВ JULIA
# ============================================================================
println("\n# ================ 8. ЛУЧШИЕ ПРАКТИКИ ================")

println("""
ЛУЧШИЕ ПРАКТИКИ ДЛЯ ПРОЕКТОВ JULIA:

1. СТРУКТУРА ПРОЕКТА
   ✓ Используйте Pkg.generate() для создания структуры
   ✓ Держите Project.toml под версионным контролем
   ✓ Используйте Manifest.toml для воспроизводимых окружений
   ✓ Организуйте код в директории src/
   ✓ Пишите тесты в директории test/

2. ДИЗАЙН МОДУЛЕЙ
   ✓ Экспортируйте только публичный API (используйте export)
   ✓ Префиксируйте внутренние функции подчёркиванием (_)
   ✓ Используйте docstrings для всех публичных функций
   ✓ Держите модули сфокусированными и однонаправленными
   ✓ Используйте подмодули для больших проектов

3. ДИЗАЙН ТИПОВ
   ✓ Используйте неизменяемые структуры по умолчанию (лучшая производительность)
   ✓ Используйте mutable struct только когда необходимо
   ✓ Определите параметрические типы для гибкости
   ✓ Используйте абстрактные типы для иерархий типов
   ✓ Реализуйте Base.show() для пользовательских типов

4. ПРОИЗВОДИТЕЛЬНОСТЬ
   ✓ Предварительно выделяйте массивы когда возможно
   ✓ Используйте представления (@view) вместо копий
   ✓ Избегайте глобальных переменных в критичном к производительности коде
   ✓ Используйте трансляцию (.+) вместо циклов
   ✓ Типовая стабильность: функции должны возвращать согласованные типы

5. ДОКУМЕНТАЦИЯ
   ✓ Пишите docstrings используя Markdown
   ✓ Включайте примеры в docstrings
   ✓ Используйте Documenter.jl для генерации документации
   ✓ Держите README.md обновлённым
   ✓ Добавьте информацию о цитировании (CITATION.bib)

6. ТЕСТИРОВАНИЕ
   ✓ Используйте Test.jl для юнит-тестов
   ✓ Тестируйте граничные случаи и условия ошибок
   ✓ Стремитесь к высокому покрытию кода
   ✓ Используйте непрерывную интеграцию (GitHub Actions)
   ✓ Включите бенчмарки производительности

7. ВЕРСИОННЫЙ КОНТРОЛЬ
   ✓ Используйте Git для версионного контроля
   ✓ Следуйте семантическому версионированию (SemVer)
   ✓ Пишите осмысленные сообщения коммитов
   ✓ Используйте ветки для функций/исправлений
   ✓ Правильно тегируйте релизы
""")

# ============================================================================
# РАЗДЕЛ 9: ПРИМЕР ПОЛНОГО ПРОЕКТА
# ============================================================================
println("\n# ================ 9. ПРИМЕР ПОЛНОГО ПРОЕКТА ================")

println("""
ПРИМЕР СТРУКТУРЫ ПРОЕКТА ДЛЯ БИОЛОГИЧЕСКОГО МОДЕЛИРОВАНИЯ:

BiologicalModels/
├── Project.toml
├── Manifest.toml
├── README.md
├── LICENSE
├── CITATION.bib
├── src/
│   ├── BiologicalModels.jl      # Главный файл модуля
│   ├── population/              # Популяционная динамика
│   │   ├── malthus.jl
│   │   ├── logistic.jl
│   │   └── lotka_volterra.jl
│   ├── epidemiology/            # Эпидемиологические модели
│   │   ├── sir.jl
│   │   ├── seir.jl
│   │   └── parameters.jl
│   ├── physiology/              # Физиологические модели
│   │   ├── insulin_glucose.jl
│   │   └── immune_response.jl
│   └── utils/                   # Вспомогательные функции
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

ПРИМЕР PROJECT.TOML:
─────────────────────
name = "BiologicalModels"
uuid = "12345678-1234-1234-1234-123456789012"
authors = ["Ваше Имя <your.email@example.com>"]
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
# РАЗДЕЛ 10: УПРАЖНЕНИЯ
# ============================================================================
println("\n# ================ 10. УПРАЖНЕНИЯ ================")

println("""
УПРАЖНЕНИЯ ДЛЯ ЧАСТИ 4:

1. СОЗДАНИЕ МОДУЛЯ
   Создайте модуль "DataUtilities" с функциями для:
   - Загрузки данных из CSV
   - Обработки пропущенных значений
   - Базовых трансформаций данных

2. ПОЛЬЗОВАТЕЛЬСКИЙ ТИП
   Создайте структуру "DataSet", которая содержит:
   - Данные (матрица или DataFrame)
   - Имена переменных (вектор строк)
   - Метаданные (словарь)
   Реализуйте пользовательский метод show()

3. РАСШИРЕНИЕ СТАТИСТИЧЕСКОГО МОДУЛЯ
   Добавьте функции в StatisticsTools:
   - Расчёт доверительного интервала
   - Бутстрап ресэмплинг
   - Коррекция множественных сравнений

4. ВИЗУАЛИЗАЦИЯ
   Создайте функцию, которая генерирует "дашборд" график с:
   - Гистограммой
   - Box plot
   - Q-Q plot
   - Таблицей сводной статистики

5. НАСТРОЙКА ПРОЕКТА
   Создайте правильную структуру проекта Julia:
   - Инициализируйте с Pkg
   - Добавьте зависимости
   - Создайте директории src/ и test/
   - Напишите базовые тесты

6. ДОКУМЕНТАЦИЯ
   Добавьте комплексные docstrings ко всем функциям в вашем модуле
   Включите примеры и граничные случаи

7. ПРОИЗВОДИТЕЛЬНОСТЬ
   Сравните производительность разных реализаций:
   - Цикл vs трансляция
   - Предварительно выделенные vs растущие массивы
   - Представления vs копии

8. ИНТЕГРАЦИЯ
   Интегрируйте StatisticsTools с:
   - DataFrames для табличных данных
   - DifferentialEquations для анализа выхода моделей
   - Plots/Makie для визуализации
""")

# ============================================================================
# РАЗДЕЛ 11: СВОДКА
# ============================================================================
println("\n# ================ 11. СВОДКА ================")

println("""
╔══════════════════════════════════════════════════════════════════════════╗
║                         СВОДКА ЧАСТИ 4                                   ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  КЛЮЧЕВЫЕ КОНЦЕПЦИИ:                                                     ║
║  ─────────────────────                                                   ║
║  1. Организация и структура проектов Julia                               ║
║  2. Создание и организация модулей                                       ║
║  3. Определение пользовательских типов (структуры)                       ║
║  4. Реализация описательной статистики                                   ║
║  5. Статистическая визуализация                                          ║
║  6. Обзор экосистемы пакетов                                             ║
║  7. Лучшие практики разработки на Julia                                  ║
║                                                                          ║
║  СГЕНЕРИРОВАННЫЕ ФАЙЛЫ:                                                  ║
║  ────────────────                                                        ║
║  • stats_histogram.png                                                   ║
║  • stats_boxplot.png                                                     ║
║  • stats_qq.png                                                          ║
║  • stats_distribution.png                                                ║
║  • stats_distribution_compare.png                                        ║
║                                                                          ║
║  ПОЛЬЗОВАТЕЛЬСКИЙ МОДУЛЬ:                                                ║
║  ──────────────                                                          ║
║  • StatisticsTools - Комплексный статистический модуль с:                ║
║    - Тип StatisticalSummary                                              ║
║    - Функции описательной статистики                                     ║
║    - Обнаружение выбросов                                                ║
║    - Нормализация/стандартизация                                         ║
║    - Функции статистического построения графиков                         ║
║                                                                          ║
║  СЛЕДУЮЩИЕ ШАГИ (ЧАСТЬ 5):                                               ║
║  ────────────────────                                                    ║
║  • Прикладное биологическое моделирование:                               ║
║    - Динамика инсулин-глюкоза                                            ║
║    - Моделирование иммунного ответа                                      ║
║    - Эпидемиологические модели (SIR, SEIR)                               ║
║    - Фармакокинетические модели                                          ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

println("\n=== Конец Основ Julia Часть 4 ===")
println("Теперь у вас есть основа для создания профессиональных проектов на Julia!")
println("В Части 5 мы применим эти навыки к реальным задачам биологического моделирования.")