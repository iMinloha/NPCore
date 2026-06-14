#ifndef NPCORE_CORE_EXPORT_H
#define NPCORE_CORE_EXPORT_H

// =================================[DLL Export/Import]================================
// NPCORE_BUILD_DLL — defined when building NPCore as a shared library
// NPCORE_USE_DLL   — defined when linking against NPCore DLL
// Neither          — static library (no-op)

#if defined(NPCORE_BUILD_DLL)
    #define NPCORE_API __declspec(dllexport)
#elif defined(NPCORE_USE_DLL)
    #define NPCORE_API __declspec(dllimport)
#else
    #define NPCORE_API
#endif

#endif // NPCORE_CORE_EXPORT_H
