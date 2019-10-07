#include <list>
#include <chrono>


#include <mrpt/gui/CDisplayWindow3D.h>  // For visualization windows
#include <mrpt/obs/CRawlog.h>
#include <mrpt/serialization/CArchive.h>
#include <mrpt/system/filesystem.h>
#include <mrpt/system/os.h>
#include <mrpt/vision/CVideoFileWriter.h>
#include <mrpt/vision/tracking.h>

using namespace mrpt;
using namespace std;
using namespace cv;
using namespace mrpt::obs;
using namespace mrpt::img;
using namespace mrpt::config;
using namespace mrpt::system;
using namespace mrpt::vision;
using namespace mrpt::poses;
using namespace mrpt::serialization;
using namespace mrpt::gui;
mrpt::gui::CDisplayWindow3D::Ptr win;


int main()
{
  int init_frame_id=1;
  win = mrpt::gui::CDisplayWindow3D::Create("Tracked features", 800, 600);

  TCamera cameraParams;  // For now, will only hold the image resolution on
	// the arrive of the first frame.
  TKeyPointList trackedFeats;
	bool DO_HIST_EQUALIZE_IN_GRAYSCALE = false;
  bool SHOW_FEAT_IDS = true;
	bool SHOW_RESPONSES = true;
	bool SHOW_SCALE = true;
	bool SHOW_FEAT_TRACKS = true;
  const double MAX_FPS = 5000;  // 5.0;  // Hz (to slow down visualization).


  CGenericFeatureTrackerAutoPtr tracker;
  // "CFeatureTracker_KL" is by far the most robust implementation for now:
	tracker = CGenericFeatureTrackerAutoPtr(new CFeatureTracker_KL);
    // Set of parameters common to any tracker implementation:
	// -------------------------------------------------------------
	// To see all the existing params and documentation, see
	// mrpt::vision::CGenericFeatureTracker
	// automatically remove out-of-image and badly tracked features

  tracker->enableTimeLogger(true);  // Do time profiling.
	tracker->extra_params["remove_lost_features"] = 1;

	// track, AND ALSO, add new features
	tracker->extra_params["add_new_features"] = 1;
	tracker->extra_params["desired_num_features_adapt"] = 500;
	tracker->extra_params["add_new_feat_min_separation"] = 30;
	tracker->extra_params["add_new_feat_max_features"] = 350;
    // FAST9,10,12:
	tracker->extra_params["add_new_features_FAST_version"] = 10;

	// Don't use patches
	tracker->extra_params["add_new_feat_patch_size"] = 0;
	tracker->extra_params["update_patches_every"] = 0;

	// KLT-response to ensure good features:
	tracker->extra_params["minimum_KLT_response_to_add"] = 70;
	tracker->extra_params["check_KLT_response_every"] = 1;
	tracker->extra_params["minimum_KLT_response"] = 30;
	tracker->extra_params["KLT_response_half_win"] = 8;

	// Specific params for "CFeatureTracker_KL"
	// ------------------------------------------------------
	tracker->extra_params["window_width"] = 15;
	tracker->extra_params["window_height"] = 15;
	tracker->extra_params["LK_levels"] = 3;



	// tracker->extra_params["LK_max_iters"] = 10;
	// tracker->extra_params["LK_epsilon"] = 0.1;
	// tracker->extra_params["LK_max_tracking_error"] = 150;
    // --------------------------------
	// The main loop
	// --------------------------------
	CImage previous_image;

	TSequenceFeatureObservations feat_track_history;
	bool save_tracked_history =true;  // Dump feat_track_history to a file at the end
  bool hasResolution = false;
	TCameraPoseID curCamPoseId = 0;

	cout << endl << "TO END THE PROGRAM: Close the window.\n";

	mrpt::opengl::COpenGLViewport::Ptr gl_view;
	{
		mrpt::opengl::COpenGLScene::Ptr scene = win->get3DSceneAndLock();
		gl_view = scene->getViewport("main");
		win->unlockAccess3DScene();
	}
    // Aux data for drawing the recent track of features:
	static const size_t FEATS_TRACK_LEN = 10;
	std::map<TFeatureID, std::list<TPixelCoord>> feat_tracks;
  std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
	// infinite loop, until we close the win:
	while (win->isOpen()&&init_frame_id<5000)
	{
        CImage theImg;  // The grabbed image:
        char file[200];
        sprintf(file, "/media/seif/ssd_workspace/data/Data_collected_for _paper_GlobalSIP/6th_f_2/ZED/sequence/left/left%06d.png", init_frame_id++);
        std::string filename = std::string(file);
        //auto o = std::dynamic_pointer_cast<CObservationImage>(cv::imread(filename, cv::IMREAD_COLOR));
		    //theImg = CImage(cv::imread(filename, cv::IMREAD_COLOR),DEEP_COPY);
        theImg.loadFromFile(file,0);
        //theImg.forceLoad();
        if (!hasResolution)
		        {
			           hasResolution = true;
			           // cameraParams.scaleToResolution()...
			           cameraParams.ncols = theImg.getWidth();
			           cameraParams.nrows = theImg.getHeight();
		         }


        if (init_frame_id > 2)  // we need "previous_image" to be valid.
		      {
			         // This single call makes: detection, tracking, recalculation of
			        // KLT_response, etc.
			        tracker->trackFeatures(previous_image, theImg, trackedFeats);
		      }
        // Save the image for the next step:
		previous_image = theImg;

		// Save history of feature observations:
		tracker->getProfiler().enter("Save history");
        for (auto& f : trackedFeats)
		{
			const TPixelCoordf pxRaw(f.pt.x, f.pt.y);
			TPixelCoordf pxUndist;
			// mrpt::vision::pinhole::undistort_point(pxRaw,pxUndist,
			// cameraParams);
			pxUndist = pxRaw;

			feat_track_history.push_back(
				TFeatureObservation(f.ID, curCamPoseId, pxUndist));
		}
        tracker->getProfiler().leave("Save history");

		// now that we're done with the image, we can directly write onto it
		//  for the display
		// ----------------------------------------------------------------
		if (DO_HIST_EQUALIZE_IN_GRAYSCALE && !theImg.isColor())
			theImg.equalizeHist(theImg);

		tracker->getProfiler().enter("Display");

		// Convert to color so we can draw color marks, etc.
		{
			mrpt::system::CTimeLoggerEntry tle(
				tracker->getProfiler(), "Display.to_color");

			theImg = theImg.colorImage();
		}

		double extra_tim_to_wait = 0;

		{  // FPS:
			static CTicTac tictac;
			const double T = tictac.Tac();
			tictac.Tic();
			const double fps = 1.0 / (std::max(1e-5, T));

			const int current_adapt_thres =
				tracker->getDetectorAdaptiveThreshold();

			mrpt::system::CTimeLoggerEntry tle(
				tracker->getProfiler(), "Display.textOut");

			theImg.selectTextFont("6x13B");
			theImg.textOut(
				3, 3, format("FPS: %.03f Hz", fps), TColor(200, 200, 0));
			theImg.textOut(
				3, 22,
				format(
					"# feats: %u - Adaptive threshold: %i",
					(unsigned int)trackedFeats.size(), current_adapt_thres),
				TColor(200, 200, 0));

			theImg.textOut(
				3, 41,
				format(
					"# raw feats: %u - Removed: %u",
					(unsigned int)tracker->last_execution_extra_info
						.raw_FAST_feats_detected,
					(unsigned int)
						tracker->last_execution_extra_info.num_deleted_feats),
				TColor(200, 200, 0));

			extra_tim_to_wait = 1.0 / MAX_FPS - 1.0 / fps;
		}

		// Draw feature tracks
		if (SHOW_FEAT_TRACKS)
		{
			// Update new feature coords:
			mrpt::system::CTimeLoggerEntry tle(
			tracker->getProfiler(), "Display.drawFeatureTracks");

			std::set<TFeatureID> observed_IDs;

			for (const auto& ft : trackedFeats)
			{
				std::list<TPixelCoord>& seq = feat_tracks[ft.ID];

				observed_IDs.insert(ft.ID);

				if (seq.size() >= FEATS_TRACK_LEN) seq.erase(seq.begin());
				seq.push_back(ft.pt);

				// Draw:
				if (seq.size() > 1)
				{
					const auto it_end = seq.end();

					auto it = seq.begin();
					auto it_prev = it++;

					for (; it != it_end; ++it)
					{
						theImg.line(
							it_prev->x, it_prev->y, it->x, it->y,
							TColor(190, 190, 190));
						it_prev = it;
					}
				}
			}

			// Purge old data:
			for (auto it = feat_tracks.begin(); it != feat_tracks.end();)
			{
				if (observed_IDs.find(it->first) == observed_IDs.end())
				{
					auto next_it = it;
					next_it++;
					feat_tracks.erase(it);
					it = next_it;
				}
				else
					++it;
			}
		}

		// Draw Tracked feats:
		{
			mrpt::system::CTimeLoggerEntry tle(
				tracker->getProfiler(), "Display.drawFeatures");

			theImg.selectTextFont("5x7");
			theImg.drawFeatures(
				trackedFeats, TColor::blue(), SHOW_FEAT_IDS, SHOW_RESPONSES,
				SHOW_SCALE, '+' /* marker */);
		}

		// Update window:
		{
			mrpt::system::CTimeLoggerEntry tle(
				tracker->getProfiler(), "Display.updateView");

			win->get3DSceneAndLock();
			gl_view->setImageView(theImg);
			win->unlockAccess3DScene();
			win->repaint();
		}

		tracker->getProfiler().leave("Display");
        if (extra_tim_to_wait > 0)
			std::this_thread::sleep_for(
				std::chrono::duration<double, std::milli>(
					1000.0 * extra_tim_to_wait));


        }
        std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        double total_track= std::chrono::duration_cast<std::chrono::duration<double> >(end_time - start_time).count();
        cout<<total_track<<endl;

    return 0;
}
