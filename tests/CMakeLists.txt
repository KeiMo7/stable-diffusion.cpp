find_package(Threads REQUIRED)

add_executable(sd-step-budget-tests
    step_budget_test.cpp
    ../src/runtime/step_budget.cpp
)

target_include_directories(sd-step-budget-tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_compile_features(sd-step-budget-tests PRIVATE cxx_std_17)

if (NOT MSVC)
    target_compile_options(sd-step-budget-tests PRIVATE -Wall -Wextra)
endif()

target_link_libraries(sd-step-budget-tests PRIVATE Threads::Threads)

add_test(NAME step_budget COMMAND sd-step-budget-tests)
