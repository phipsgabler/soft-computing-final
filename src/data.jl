using DataFrames
using CSV
using CategoricalArrays

# Glass identification:
# 1. Id number: 1 to 214
# 2. RI: refractive index
# 3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# 4. Mg: Magnesium
# 5. Al: Aluminum
# 6. Si: Silicon
# 7. K: Potassium
# 8. Ca: Calcium
# 9. Ba: Barium
# 10. Fe: Iron
# 11. Type of glass: (class attribute)
# -- 1 building_windows_float_processed
# -- 2 building_windows_non_float_processed
# -- 3 vehicle_windows_float_processed
# -- 4 vehicle_windows_non_float_processed (none in this database)
# -- 5 containers
# -- 6 tableware
# -- 7 headlamps

glass = CSV.read("../data/glass-identification.data",
                 header = [:Id, :Ri, :Na, :Mg, :Al, :Si, :K, :Ca, :Ba, :Fe, :Type],
                 transforms = Dict("Type" => string))
conversions = map(=>, string.(1:7),
                  ["building_windows_float_processed",
                   "building_windows_non_float_processed",
                   "vehicle_windows_float_processed",
                   "vehicle_windows_non_float_processed",
                   "containers",
                   "tableware",
                   "headlamps"])
recode!(glass[:Type], conversions...)
