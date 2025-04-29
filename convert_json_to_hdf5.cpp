#include <iostream>
#include <string>
#include <vector>
#include "json.hpp" // For nlohmann::json
#include "H5Cpp.h"  // For HDF5 C++ API
#include <fstream>

using json = nlohmann::json;
using namespace H5;

int main() {

    std::ifstream input_file("/mnt/pny_4tb/code/flask_stuff/pokemon_card_search_engine/complete_set_of_pokemon_card_pairs.json");
    nlohmann::json json_data;
    input_file >> json_data;
    input_file.close();

    // Convert JSON data to vector of strings for HDF5
    std::vector<std::string> string_data;
    for (const auto& pair : json_data) {
        string_data.push_back(pair[0]);
        string_data.push_back(pair[1]);
    }

    // Create an HDF5 file
    H5File file("pokemon_card_pairs.h5", H5F_ACC_TRUNC);

    // Create a dataspace for the dataset
    hsize_t dims[2] = {json_data.size(), 2};
    DataSpace dataspace(2, dims);

    // Create a string datatype
    StrType strdatatype(PredType::C_S1, H5T_VARIABLE);

    // Create the dataset
    DataSet dataset = file.createDataSet("complete_set_of_pokemon_card_pairs", strdatatype, dataspace);

    // Write the data to the dataset
    dataset.write(string_data.data(), strdatatype);

    // Close the dataset, dataspace, and file
    dataset.close();
    dataspace.close();
    file.close();

    std::cout << "Converted JSON list of string pairs to HDF5 format." << std::endl;

    return 0;
}