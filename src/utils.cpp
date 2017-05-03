#include <iostream>
#include <fstream>
#include "../include/utils.h"

using namespace std;
void display_json(const std::map<std::string, double>& info,
        const std::vector<cv::Point2f>& ptpairs, string& json_file) {
    cout << "[display_json] JSON output >>>>>> " << std::endl;

    // Local variables
    Json::Value jsonRoot;
    Json::Value jsonArray;
    Json::StyledWriter styledWriter;
    std::ofstream fileStream;
    std::map<std::string, double>::const_iterator i;

    // Create empty array
    jsonRoot["matched-points-centerpoint"] = Json::Value(Json::arrayValue);

    // Add info to json object
    for (i = info.begin(); i != info.end(); i++) {
        jsonRoot[i->first] = i->second;
    }

    // Insert key points to json array
    for (size_t i = 0; i < ptpairs.size(); ++i) {
        jsonArray["x"] = (int) (ptpairs[i].x);
        jsonArray["y"] = (int) (ptpairs[i].y);
        jsonRoot["matched-points-centerpoint"].append(jsonArray);
    }

    // Output json to std output
    std::string jsonString = styledWriter.write(jsonRoot);
    std::cout << jsonString;

    // Write info to json file
    if(json_file.empty()) return;

    fileStream.open(json_file);
    if(fileStream.is_open()) {
        fileStream << styledWriter.write(jsonRoot);
        std::cout << "JSON file is saved to " << json_file << std::endl;
    }
    else {
        cerr << "Error in open json file!!" << endl;
    }

    fileStream.close();

    std::cout << "[display_json] <<<<<< " << std::endl;
}
