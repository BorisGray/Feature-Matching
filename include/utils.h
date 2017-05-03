#ifndef __utils_H__
#define __utils_H__

#include <map>
#include <vector>
#include <string>
#include "opencv2/core.hpp"
#include "./json/json.h"

#if PERF_COUNT_VERBOSE >= 2
#define PERF_COUNT_START(FUNCT_NAME) \
    char *funct_name = FUNCT_NAME; \
    double elapsed_time_sec; \
    timespec time_funct_start, time_funct_end, time_sub_start, time_sub_end; \
    clock_gettime( CLOCK_MONOTONIC, &time_funct_start ); \
    time_sub_start = time_funct_start; \
    time_sub_end = time_funct_start;

#define PERF_COUNT_END                                                    \
  clock_gettime(CLOCK_MONOTONIC, &time_funct_end);                        \
  elapsed_time_sec =                                                      \
      (time_funct_end.tv_sec - time_funct_start.tv_sec) * 1.0 +           \
      (time_funct_end.tv_nsec - time_funct_start.tv_nsec) / 1000000000.0; \
  printf(PERF_COUNT_REPORT_STR, funct_name, "total", elapsed_time_sec);
#else
#define PERF_COUNT_START(FUNCT_NAME)
#define PERF_COUNT_END
#endif


// Stringify common types such as int, double and others.
template <typename T>
inline std::string to_string(const T& x) {
  std::stringstream oss;
  oss << x;
  return oss.str();
}

void display_json(const std::map<std::string, double>& info, const std::vector<cv::Point2f>& ptpairs, std::string& json_file);

void draw_keypoints(cv::Mat& img, const std::vector<cv::KeyPoint>& kpts);

#endif // __utils_H__
