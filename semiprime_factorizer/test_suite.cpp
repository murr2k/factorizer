/**
 * Test Suite for Semiprime Factorizer v3.0.0
 * Comprehensive testing with various semiprime types
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <iomanip>

using namespace std;
using namespace chrono;

struct TestCase {
    string number;
    string factor1;
    string factor2;
    string description;
};

int main() {
    cout << "==================================================" << endl;
    cout << "  Semiprime Factorizer v3.0.0 - Test Suite" << endl;
    cout << "==================================================" << endl << endl;
    
    vector<TestCase> tests = {
        // Small semiprimes
        {"143", "11", "13", "Small semiprime (8 bits)"},
        {"1234567", "127", "9721", "Medium semiprime (21 bits)"},
        
        // Balanced factors
        {"1000000016000000063", "1000000007", "1000000009", "Balanced factors (60 bits)"},
        
        // Original test cases from v2.x that failed
        {"15482526220500967432610341", "1804166129797", "8581541336353", "26-digit case (84 bits)"},
        {"46095142970451885947574139", "6543896059637", "7043990697647", "86-bit balanced factors"},
        {"71074534431598456802573371", "7168379511841", "9915007194331", "86-bit case 2"},
        
        // User-provided test cases
        {"139789207152250802634791", "206082712973", "678316027267", "77-bit user case 1"},
        {"11690674751274331636209053", "2494580829527", "4686428522539", "84-bit user case 2"},
        
        // Edge cases
        {"4", "2", "2", "Perfect square of prime"},
        {"123456789123456789", "3", "41152263041152263", "One small factor"},
    };
    
    int passed = 0;
    int failed = 0;
    double total_time = 0;
    
    cout << "Running " << tests.size() << " test cases..." << endl << endl;
    
    for (const auto& test : tests) {
        cout << "Test: " << test.description << endl;
        cout << "Number: " << test.number << endl;
        
        // Run factorizer
        string cmd = "./semiprime_factor " + test.number + " 2>&1";
        auto start = high_resolution_clock::now();
        FILE* pipe = popen(cmd.c_str(), "r");
        
        if (!pipe) {
            cout << "❌ FAILED: Could not run factorizer" << endl;
            failed++;
            continue;
        }
        
        // Read output
        char buffer[256];
        string output;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }
        pclose(pipe);
        
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        double time_seconds = duration.count() / 1000.0;
        total_time += time_seconds;
        
        // Check if factorization was successful
        bool success = output.find("Factorization successful!") != string::npos;
        bool correct = (output.find(test.factor1) != string::npos || 
                       output.find(test.factor2) != string::npos);
        
        if (success && correct) {
            cout << "✅ PASSED in " << fixed << setprecision(3) << time_seconds << " seconds" << endl;
            passed++;
        } else {
            cout << "❌ FAILED" << endl;
            if (!success) cout << "   Factorization failed" << endl;
            if (!correct) cout << "   Incorrect factors" << endl;
            failed++;
        }
        
        cout << endl;
    }
    
    // Summary
    cout << "==================================================" << endl;
    cout << "                 TEST SUMMARY" << endl;
    cout << "==================================================" << endl;
    cout << "Total tests: " << tests.size() << endl;
    cout << "Passed: " << passed << " (" << (passed * 100.0 / tests.size()) << "%)" << endl;
    cout << "Failed: " << failed << endl;
    cout << "Average time: " << fixed << setprecision(3) << (total_time / tests.size()) << " seconds" << endl;
    cout << "Total time: " << total_time << " seconds" << endl;
    
    if (failed == 0) {
        cout << endl << "🎉 ALL TESTS PASSED! 🎉" << endl;
    }
    
    return failed;
}