#ifndef __FEATURE_MATCHING_H__
#define __FEATURE_MATCHING_H__

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "utils.h"

using namespace cv;
using namespace std;

namespace FEATURE_MATCHING_NS {

  typedef vector<KeyPoint> KeyPoints;

  enum DetectorExractorPolicy {
    SURF_POLICY 	= 0,
    SIFT_POLICY 	= 1,
    AKAZE_POLICY 	= 2,
  };

  enum DescriptorMatcherPolicy {
    FLANN_BASED_POLICY         = 0,
    BRUTE_FORCE_HAMMING_POLICY = 1,
  };

  enum CalibEstimatePolicy {
    RANSAC_POLICY	 = 8,
    LMEDS_POLICY     = 4,
  };

  struct FeatureMatchingTiming {

    FeatureMatchingTiming() {
      detector_tm   = 0.0;
      descriptor_tm = 0.0;
      matcher_tm    = 0.0;
      homography_tm = 0.0;
      transform_tm  = 0.0;
    }

    double detector_tm  ;
    double descriptor_tm;
    double matcher_tm   ;
    double homography_tm;
    double transform_tm ;
  };

  struct FeatureMatchingOptions {
    FeatureMatchingOptions() {
      detectorExractorPolicy    = SURF_POLICY;
      descriptorMatcherPolicy   = FLANN_BASED_POLICY;
      calibEstimatePolicy       = RANSAC_POLICY;

      min_hessian  = 400;
      max_distance = 0;
      min_distance = 100;
    }

    FeatureMatchingOptions& operator=(const FeatureMatchingOptions& rhs) {
      if  ((this) == &rhs ) return *this;
      detectorExractorPolicy = rhs.detectorExractorPolicy;
      descriptorMatcherPolicy = rhs.descriptorMatcherPolicy;
      calibEstimatePolicy = rhs.calibEstimatePolicy;
      min_hessian = rhs.min_hessian;
      max_distance = rhs.max_distance;
      min_distance = rhs.min_distance;
    }

    FeatureMatchingOptions(const FeatureMatchingOptions& rhs) {
      detectorExractorPolicy = rhs.detectorExractorPolicy;
      descriptorMatcherPolicy = rhs.descriptorMatcherPolicy;
      calibEstimatePolicy = rhs.calibEstimatePolicy;
      min_hessian = rhs.min_hessian;
      max_distance = rhs.max_distance;
      min_distance = rhs.min_distance;
    }

    DetectorExractorPolicy    detectorExractorPolicy;
    DescriptorMatcherPolicy   descriptorMatcherPolicy;
    CalibEstimatePolicy       calibEstimatePolicy;

    int     min_hessian;
    double  max_distance;
    double  min_distance;

  };

  class FeatureMatching
  {
  public:

    FeatureMatching(const char* img_obj_name, const char* img_scn_name,
                    const FeatureMatchingOptions& feat_match_opts, bool js_flag = true);
    FeatureMatching(const Rect& rect_obj, const char* img_scn_name,
                    const FeatureMatchingOptions& feat_match_opts, bool js_flag = true);

    ~FeatureMatching();

    bool process();

  private:

    void init(void);

    void detect(void);
    void compute(void);

    void match(void);

    void calcMaxAndMinDist(double& max_dist, double& min_dist);
    void goodMatchs();

    void findHompgraphy(Mat& hg);
    void transform(const Mat& hg);

    void calcCenterPoint(const vector<Point2f>& poly_line);
    bool verifyPoints();

    void drawRectCenter();
    void showComputationTimes() const;

    void save_result();
  private:
    bool                    json_flag;
    FeatureMatchingOptions  feat_match_opts;
    FeatureMatchingTiming   timing;

    Mat img_object;
    Mat img_scene;

    Ptr<Feature2D>          detector;
    Ptr<DescriptorMatcher>  matcher;

    vector< DMatch > matches;
    vector< DMatch > good_matches;

    Mat img_matches;

    KeyPoints keypoints_object;
    KeyPoints keypoints_scene;

    Mat descriptors_object;
    Mat descriptors_scene;

    vector< Point2f > scene_corners;
    Point2f           cent_pnt;

  private:
    FeatureMatching& operator=(const FeatureMatching&);
    FeatureMatching(const FeatureMatching&);
  };

} // namespace FEATURE_MATCHING_NS

#endif // __FEATURE_MATCHING_H__
