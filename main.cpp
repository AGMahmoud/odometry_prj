#include <list>
#include <chrono>


#include <mrpt/gui/CDisplayWindow3D.h>  // For visualization windows
#include <mrpt/obs/CRawlog.h>
#include <mrpt/serialization/CArchive.h>
#include <mrpt/system/filesystem.h>
#include <mrpt/system/os.h>
#include <mrpt/vision/CVideoFileWriter.h>
#include <mrpt/vision/tracking.h>
#include <mrpt/obs/CObservationStereoImages.h>
#include <mrpt/img/TStereoCamera.h>
#include <mrpt/img/TCamera.h>
#include <mrpt/config/CConfigFileBase.h>
#include <mrpt/math/TPose3DQuat.h>

using namespace mrpt;
using namespace std;
using namespace cv;
using namespace mrpt::obs;
using namespace mrpt::math;
using namespace mrpt::img;
using namespace mrpt::config;
using namespace mrpt::system;
using namespace mrpt::vision;
using namespace mrpt::poses;
using namespace mrpt::serialization;
using namespace mrpt::gui;
mrpt::gui::CDisplayWindow3D::Ptr win1,win2;
int main()
{
  char x;
  int init_frame_id=1;
  bool hasResolution = false;
  TCamera cameraParams;  // For now, will only hold the image resolution on


  // creating stero camera and load the camera parameters
  CObservationStereoImages Zed_stereo_obs;
  CPose3D sensor_pose;
  TPose3DQuat rigt_cam_pose =  TPose3DQuat(63.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0) ;
  TStereoCamera out_params;
  Zed_stereo_obs.getSensorPose(sensor_pose);
  Zed_stereo_obs.leftCamera.setIntrinsicParamsFromValues (698.9, 694.89, 645.562, 343.643);
  Zed_stereo_obs.rightCamera.setIntrinsicParamsFromValues (698.9, 694.89, 645.562, 343.643);
  Zed_stereo_obs.leftCamera.setDistortionParamsFromValues (0.0, 0.0, 0.0, 0.0, 0.0);
  Zed_stereo_obs.rightCamera.setDistortionParamsFromValues (0.0, 0.0, 0.0, 0.0, 0.0);
  Zed_stereo_obs.rightCameraPose = rigt_cam_pose;
  Zed_stereo_obs.getStereoCameraParams(out_params);
  cout<<out_params.dumpAsText();
  cin>>x;
  win1 = mrpt::gui::CDisplayWindow3D::Create("Tracked features", 800, 600);
  win2 = mrpt::gui::CDisplayWindow3D::Create("Tracked features", 800, 600);

  CImage previous_image;
  cout << endl << "TO END THE PROGRAM: Close the window.\n";
  mrpt::opengl::COpenGLViewport::Ptr gl_view1;
  {
    mrpt::opengl::COpenGLScene::Ptr scene1 = win1->get3DSceneAndLock();
    gl_view1 = scene1->getViewport("main");
    win1->unlockAccess3DScene();
  }
  mrpt::opengl::COpenGLViewport::Ptr gl_view2;
  {
    mrpt::opengl::COpenGLScene::Ptr scene2 = win2->get3DSceneAndLock();
    gl_view2 = scene2->getViewport("main2");
    win2->unlockAccess3DScene();
  }
  while (win->isOpen()&&init_frame_id<5000)
	{
        CImage theImg;  // The grabbed image:
        char file[200];
        sprintf(file, "/media/seif/ssd_workspace/data/Data_collected_for _paper_GlobalSIP/6th_f_2/ZED/sequence/left/left%06d.png", init_frame_id++);
        std::string filename = std::string(file);
        //auto o = std::dynamic_pointer_cast<CObservationImage>(cv::imread(filename, cv::IMREAD_COLOR));
		    //theImg = CImage(cv::imread(filename, cv::IMREAD_COLOR),DEEP_COPY);
        theImg.loadFromFile(file,1);
        //theImg.forceLoad();
        if (!hasResolution)
		        {
			           hasResolution = true;
			           // cameraParams.scaleToResolution()...
			           cameraParams.ncols = theImg.getWidth();
			           cameraParams.nrows = theImg.getHeight();
		         }
             static CTicTac tictac;
             const double T = tictac.Tac();
             tictac.Tic();
             const double fps = 1.0 / (std::max(1e-5, T));

             theImg.selectTextFont("6x13B");
             theImg.textOut(3, 3, format("FPS: %.03f Hz", fps), TColor(200, 200, 0));

             win1->get3DSceneAndLock();
       			 gl_view1->setImageView(theImg);
       			 win1->unlockAccess3DScene();
       			 win1->repaint();

    }
  }
