#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace zoo
{
enum class OneEnum
{
    TEST,
    BYE
};

class Other
{
    unsigned char foo;
    OneEnum bar;
};
}

struct Nested
{
    float a[10];
    std::string name;
    std::array<float, 32> vals;
};

struct TopStruct
{
    double a;
    std::vector<int> b;
    bool c;
    std::unique_ptr<Nested> d;
    std::vector<Nested> e;
    long unsigned int f;
    void *g;
    std::uint64_t h;
    zoo::Other other;
};

void func()
{
    // g++ -Wall -c -g -fno-eliminate-unused-debug-types -o types.o types.cpp
    // Not the magic option :). Otherwise, you MUST use the type
    if constexpr (false)
    {
        TopStruct instance;
    }
}