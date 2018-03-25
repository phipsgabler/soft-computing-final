using DataFrames
using CSV
using CategoricalArrays

datafile(filename) = joinpath(Pkg.dir("SoftComputingFinal"), "data", filename)

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

function load_glass()
    glass = CSV.read(datafile("glass-identification.data"),
                     header = [:id, :ri, :na, :mg, :al, :si, :k, :ca, :ba, :fe, :type],
                     types = Dict(11 => String))
    
    categorical!(glass, :type)
    conversions = map(=>, string.(1:7),
                      ["building_windows_float_processed",
                       "building_windows_non_float_processed",
                       "vehicle_windows_float_processed",
                       "vehicle_windows_non_float_processed",
                       "containers",
                       "tableware",
                       "headlamps"])
    recode!(glass[:type], conversions...)
    
    delete!(glass, :id)

    return glass
end




# Ionosphere
# First 34 are continuous values
# The 35th attribute is either "good" or "bad" evidence

function load_ionosphere()
    ionosphere = CSV.read(datafile("ionosphere.data"),
                          header = [:id, [Symbol("x", i) for i = 1:34]..., :evidence],
                          types = Dict(2 => Float64, 3 => Float64))
    recode!(ionosphere[:evidence], "g" => "good", "b" => "bad")
    delete!(ionosphere, :id)

    return ionosphere
end





# Image segmentation
# 19 continuous attributes:
#     1.  region-centroid-col:  the column of the center pixel of the region.
#     2.  region-centroid-row:  the row of the center pixel of the region.
#     3.  region-pixel-count:  the number of pixels in a region = 9.
#     4.  short-line-density-5:  the results of a line extractoin algorithm that 
#          counts how many lines of length 5 (any orientation) with
#          low contrast, less than or equal to 5, go through the region.
#     5.  short-line-density-2:  same as short-line-density-5 but counts lines
#          of high contrast, greater than 5.
#     6.  vedge-mean:  measure the contrast of horizontally
#          adjacent pixels in the region.  There are 6, the mean and 
#          standard deviation are given.  This attribute is used as
#         a vertical edge detector.
#     7.  vegde-sd:  (see 6)
#     8.  hedge-mean:  measures the contrast of vertically adjacent
#           pixels. Used for horizontal line detection. 
#     9.  hedge-sd: (see 8).
#     10. intensity-mean:  the average over the region of (R + G + B)/3
#     11. rawred-mean: the average over the region of the R value.
#     12. rawblue-mean: the average over the region of the B value.
#     13. rawgreen-mean: the average over the region of the G value.
#     14. exred-mean: measure the excess red:  (2R - (G + B))
#     15. exblue-mean: measure the excess blue:  (2B - (G + R))
#     16. exgreen-mean: measure the excess green:  (2G - (R + B))
#     17. value-mean:  3-d nonlinear transformation
#          of RGB. (Algorithm can be found in Foley and VanDam, Fundamentalsa
#          of Interactive Computer Graphics)
#     18. saturatoin-mean:  (see 17)
#     19. hue-mean:  (see 17)
# Class Distribution: 
#    Classes:  brickface, sky, foliage, cement, window, path, grass.

function load_segmentation()
    segmentation = CSV.read(datafile("image-segmentation.data"),
                            header = [:id,
                                      :region_centroid_col,
                                      :region_centroid_row,
                                      :region_pixel_count,
                                      :short_line_density_5,
                                      :short_line_density_2,
                                      :vedge_mean,
                                      :vedge_sd,
                                      :hedge_mean,
                                      :hedge_sd,
                                      :intensity_mean,
                                      :rawred_mean,
                                      :rawblue_mean,
                                      :rawgreen_mean,
                                      :exred_mean,
                                      :exblue_mean,
                                      :exgreen_mean,
                                      :value_mean,
                                      :saturation_mean,
                                      :hue_mean,
                                      :image],
                            types = Dict(4 => Float64),
                            transforms = Dict(21 => lowercase ∘ string))
    categorical!(segmentation, :image)
    delete!(segmentation, :id)

    return segmentation
end


# Linearly separable 2D data set from https://github.com/cuekoo/Binary-classification-dataset

function load_testdata()
    testdata = CSV.read(datafile("testdata.data"),
                        header = [:x, :y, :class])
    categorical!(testdata, :class)
end



function check_loading(df)
    display(names(df))
    display(eltypes(df))
    display(levels(df[end]))
end
