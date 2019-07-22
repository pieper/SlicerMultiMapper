cmake_minimum_required(VERSION 3.13.4)

project(MultiMapper)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "http://slicer.org/slicerWiki/index.php/Documentation/Nightly/Extensions/MultiMapper")
set(EXTENSION_CATEGORY "Quantification")
set(EXTENSION_CONTRIBUTORS "Steve Pieper (isomics, Inc.)")
set(EXTENSION_DESCRIPTION "
Tools for creating parametric maps from multidimensional MRI")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/MultiMapper.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/MultiMapper/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a space separated string, a list or 'NA' if any

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(MultiMapper)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
