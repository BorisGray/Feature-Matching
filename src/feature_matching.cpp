#include <stdio.h>
#include <iostream>
#include "opencv2/xfeatures2d.hpp"
#include "./include/feature_matching.h"

using namespace cv::xfeatures2d;
using namespace FEATURE_MATCHING_NS;

#define __DEBUG__
#define EU_DSTC_FACTOR 4
#define VERIFY_DATA(imgObject, imgScene) \
  if( !imgObject.data || !imgScene.data )  \
{   \
  cout<< " --(!) Error reading images "<<endl;  \
  return;   \
  }

FeatureMatching::FeatureMatching(const char* img_obj_name, const char* img_scn_name,
                                 const FeatureMatchingOptions& feat_mat_opts, bool js_flag)
  :   feat_match_opts(feat_mat_opts), json_flag(js_flag) {

  CV_Assert(img_scn_name && img_obj_name);

  img_object = imread( img_obj_name, CV_LOAD_IMAGE_GRAYSCALE );
  img_scene  = imread( img_scn_name, CV_LOAD_IMAGE_GRAYSCALE );

  VERIFY_DATA(img_object, img_scene);

  init();
}

FeatureMatching::FeatureMatching(const Rect& rect_obj, const char* img_scn_name,
                                 const FeatureMatchingOptions& feat_mat_opts, bool js_flag )
  :   feat_match_opts(feat_mat_opts), json_flag(js_flag) {

  CV_Assert(img_scn_name);

  img_scene  = imread( img_scn_name, CV_LOAD_IMAGE_GRAYSCALE );
  if( !img_scene.data )
    {
      cout << " --(!) Error reading image scene "<<endl;
      return;   \
    }

  img_object = img_scene(rect_obj).clone();
  if( !img_object.data )
    {
      cout << " --(!) Error reading images object "<<endl;
      return;   \
    }

#ifdef __DEBUG__
  imshow( "object", img_object );
  //    imshow( "scene", img_scene );
  waitKey(3000);
#endif
  init();
}

void FeatureMatching::init()
{
  if (SURF_POLICY == feat_match_opts.detectorExractorPolicy)
    detector = SURF::create(feat_match_opts.min_hessian);
  else if (SIFT_POLICY == feat_match_opts.detectorExractorPolicy)
    detector = SIFT::create();
  else if (AKAZE_POLICY == feat_match_opts.detectorExractorPolicy)
    detector = AKAZE::create();

  CV_Assert(detector);

  if ( FLANN_BASED_POLICY == feat_match_opts.descriptorMatcherPolicy )
    matcher = DescriptorMatcher::create("FlannBased");
  else if (BRUTE_FORCE_HAMMING_POLICY == feat_match_opts.descriptorMatcherPolicy)
    matcher = DescriptorMatcher::create("BruteForce-Hamming");

  CV_Assert(matcher);
}

FeatureMatching::~FeatureMatching() {

}

bool FeatureMatching::process() {
  detect();
  compute();

  match();

  calcMaxAndMinDist(feat_match_opts.max_distance, feat_match_opts.min_distance );
  goodMatchs();

  Mat hg;
  findHompgraphy(hg);
  transform(hg);

  calcCenterPoint(this->scene_corners);
  if (!verifyPoints()) return false;

  drawRectCenter();

  save_result ();
  return true;
}

//-- Step 1: Detect the keypoints using SURF Detector
void FeatureMatching::detect(void) {

  CV_Assert(detector);

  double t1 = 0.0, t2 = 0.0;
  t1 = cv::getTickCount();

  detector->detect(this->img_object, this->keypoints_object );
  detector->detect(this->img_scene,  this->keypoints_scene );

  t2 = cv::getTickCount();
  timing.detector_tm = 1000 * (t2-t1) / cv::getTickFrequency();

#ifdef __DEBUG__
  cout <<"key points number of object: " << this->keypoints_object.size() << endl;
  cout <<"key points number of scene: " << this->keypoints_scene.size() << endl;
#endif
}

//-- Step 2: Calculate descriptors (feature vectors)
void FeatureMatching::compute(void) {

  CV_Assert(detector);

  double t1 = 0.0, t2 = 0.0;
  t1 = cv::getTickCount();

  detector->compute(this->img_object, this->keypoints_object, this->descriptors_object );
  detector->compute(this->img_scene, this->keypoints_scene, this->descriptors_scene );

  t2 = cv::getTickCount();
  timing.descriptor_tm = 1000 * (t2-t1) / cv::getTickFrequency();
}

//-- Step 3: Matching descriptor vectors using FLANN matcher
void FeatureMatching::match(void) {

  CV_Assert(matcher);

  double t1 = 0.0, t2 = 0.0;
  t1 = cv::getTickCount();

  matcher->match( this->descriptors_object, this->descriptors_scene, this->matches );
  t2 = cv::getTickCount();
  timing.matcher_tm = 1000 * (t2-t1) / cv::getTickFrequency();
}

//-- Quick calculation of max and min distances between keypoints
void FeatureMatching::calcMaxAndMinDist(double& max_dist, double& min_dist) {

  CV_Assert(!matches.empty());

  for( int i = 0; i < this->descriptors_object.rows; i++ ){
      double dist = this->matches[i].distance;
      if( dist < min_dist ) min_dist = dist;
      if( dist > max_dist ) max_dist = dist;
    }

#ifdef __DEBUG__
  cout << "number of matches pre-filtering: "<< this->matches.size()<<endl;
  cout << "-- Max dist : " << max_dist << endl;
  cout << "-- Min dist : " << min_dist << endl;
#endif
}

//-- DRAW only "good" matches (i.e. whose distance is less than 3*min_dist )
void FeatureMatching::goodMatchs() {

  for( int i = 0; i < this->descriptors_object.rows; i++ ) {
      if( this->matches[i].distance < /*3*/EU_DSTC_FACTOR * this->feat_match_opts.min_distance ) {
          this->good_matches.push_back( matches[i]); }
    }

  CV_Assert(!good_matches.empty());
#ifdef __DEBUG__
  cout<<"number of matches after filtering: "<< this->good_matches.size()<<endl;

  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  for( size_t i = 0; i < this->good_matches.size(); i++ )
    cout << "Good match[" << i << "]: keypointsObject[" << this->good_matches[i].queryIdx
         << "] -- keypointsScene[" << this->good_matches[i].trainIdx << "]" << endl;
#endif
}

///-- Step 4:  Localize the object
void FeatureMatching::findHompgraphy(Mat& homography) {
  double t1 = 0.0, t2 = 0.0;
  t1 = cv::getTickCount();

  std::vector<Point2f> obj;
  std::vector<Point2f> scene;

  for( size_t i = 0; i < this->good_matches.size(); i++ )
    {
      //-- Get the keypoints from the good matches
      obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
      scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }

  CV_Assert(!obj.empty() && !scene.empty());
  // CalibEstimateAlgo = RANSAC
  homography = findHomography( obj, scene, /*CV_RANSAC*/this->feat_match_opts.calibEstimatePolicy );

  t2 = cv::getTickCount();
  timing.homography_tm = 1000 * (t2-t1) / cv::getTickFrequency();
}

///-- Perspective transform
void FeatureMatching::transform(const Mat& hg) {
  CV_Assert(!hg.empty());

  double t1 = 0.0, t2 = 0.0;
  t1 = cv::getTickCount();

  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0);
  obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows );
  obj_corners[3] = cvPoint( 0, img_object.rows );

  //std::vector<Point2f> scene_corners(4);
  perspectiveTransform( obj_corners, this->scene_corners, hg);

  t2 = cv::getTickCount();
  timing.transform_tm = 1000 * (t2-t1) / cv::getTickFrequency();

#ifdef __DEBUG__
  for( size_t i = 0; i < obj_corners.size(); i++ )
    cout << " matched polyline " << i << ": obj_corners[x]:"
         << obj_corners[i].x << ", obj_corners[y]:" << obj_corners[i].y <<endl;
#endif
}

bool FeatureMatching::verifyPoints()
{
#ifdef __DEBUG__
  cout <<" center point of polyline: (" << this->cent_pnt.x << ", " << this->cent_pnt.y << ")" << endl;
#endif
  if (this->cent_pnt.x < 0 || this->cent_pnt.x > img_scene.cols) return false;
  if (this->cent_pnt.y < 0 || this->cent_pnt.y > img_scene.rows) return false;
  return true;
}

void FeatureMatching::calcCenterPoint(const vector<Point2f>& poly_line) {
  Moments mu;
  mu = moments( poly_line, false );
  this->cent_pnt = Point2f( mu.m10/mu.m00, mu.m01/mu.m00 );
}

//-- Draw lines between the corners (the mapped object in the scene - image_2 )
void FeatureMatching::drawRectCenter() {

/*     line( this->img_scene, this->scene_corners[0], this->scene_corners[1], Scalar(0, 255, 0), 4 );
      line( this->img_scene, this->scene_corners[1], this->scene_corners[2], Scalar( 0, 255, 0), 4 );
      line( this->img_scene, this->scene_corners[2], this->scene_corners[3], Scalar( 0, 255, 0), 4 );
      line( this->img_scene, this->scene_corners[3], this->scene_corners[0], Scalar( 0, 255, 0), 4 );
      circle(this->img_scene, cent_pnt, 2, Scalar( 0, 0, 255));
*/
  line( this->img_matches, this->scene_corners[0] + Point2f( img_object.cols, 0), this->scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  line( this->img_matches, this->scene_corners[1] + Point2f( img_object.cols, 0), this->scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( this->img_matches, this->scene_corners[2] + Point2f( img_object.cols, 0), this->scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( this->img_matches, this->scene_corners[3] + Point2f( img_object.cols, 0), this->scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  circle(this->img_matches, this->cent_pnt + Point2f( img_object.cols, 0), 5, Scalar( 0, 0, 255), 3);
#ifdef __DEBUG__
  namedWindow("Good Matches & Object detection", WINDOW_GUI_EXPANDED);
  imshow("Good Matches & Object detection", img_matches );
  imwrite("./res.jpg", img_matches);
  waitKey(5000);
#endif

}

void FeatureMatching::showComputationTimes() const {
  cout << "(*) Time of detect key points: " << timing.detector_tm << endl;
  cout << "(*) Time of compute feature descriptor: " << timing.descriptor_tm << endl;
  cout << "(*) Time of matching descriptor vectors: " << timing.matcher_tm << endl;
  cout << "(*) Time of find homography: " << timing.homography_tm << endl;
  cout << "(*) Time of perspective transform: " << timing.transform_tm << endl;
  cout << endl;
}

// Testdroid: save matching info to map, pass map to display_json()
void FeatureMatching::save_result() {

  if(json_flag) {
      std::map<std::string, double> info;
      std::string jsonFile = "./results.json";

      info.insert(std::pair<std::string, double>("number-keypoints-imgobject", keypoints_object.size()));
      info.insert(std::pair<std::string, double>("number-keypoints-imgscene", keypoints_scene.size()));
      info.insert(std::pair<std::string, double>("number-all-macthes", matches.size()));
      info.insert(std::pair<std::string, double>("number-good-macthes", good_matches.size()));
      info.insert(std::pair<std::string, double>("time-detect-keypoints", timing.detector_tm));
      info.insert(std::pair<std::string, double>("time-compute-descriptor", timing.descriptor_tm));
      info.insert(std::pair<std::string, double>("time-matching-descriptor-vectors", timing.matcher_tm));
      info.insert(std::pair<std::string, double>("time-find-homography", timing.homography_tm));
      info.insert(std::pair<std::string, double>("time-perspective-transform", timing.transform_tm));

      std::vector<Point2f> ptpaires;
      ptpaires.assign(this->scene_corners.begin(), this->scene_corners.end());
      ptpaires.push_back(this->cent_pnt);
      display_json(info, ptpaires, jsonFile);
    }
}
