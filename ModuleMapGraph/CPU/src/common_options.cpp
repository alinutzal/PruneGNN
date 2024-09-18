/***************************************************************
 * Tracking project library - L2IT
 * Trace reconstruction in LHC
 * copyright © 2022,2023 ROUGIER Charline
 * copyright © 2022,2023,2024 COLLARD Christophe
 * copyright © 2022,2023,2024 Centre National de la Recherche Scientifique
 * copyright © 2022,2023,2024 Laboratoire des 2 Infinis de Toulouse (L2IT)
 ***************************************************************/

#include "common_options.hpp"

//=====Private methods for common options===========================================


//=====Public methods for common options============================================


//-------------------------------------------------------------------------------
common_options::common_options (boost::program_options::options_description& opt)
//-------------------------------------------------------------------------------
{
  opt.add_options() ("help,h", "Produce help message");
  opt.add_options() ("loglevel,l", boost::program_options::value<size_t>()->default_value(2), "The output log level. Please set the wished number (0 = VERBOSE, 1 = "  "DEBUG, 2 = INFO, 3 = WARNING, 4 = ERROR, 5 = FATAL).");
  opt.add_options() ("response-file", boost::program_options::value<std::string>()->default_value(""), "Configuration file (response file) replacing command line options.");
}


//------------------------------------------------------------------------------------------
void common_options::add_hardware_options (boost::program_options::options_description& opt)
//------------------------------------------------------------------------------------------
{ // sequencer options
  opt.add_options() ("gpu-nb-blocks", boost::program_options::value<int>(), "Total number of blocks on graphical card (see GPU hardware specifications)");
}


//-------------------------------------------------------------------------------------------
void common_options::add_sequencer_options (boost::program_options::options_description& opt)
//-------------------------------------------------------------------------------------------
{ // sequencer options
  opt.add_options() ("events,n", boost::program_options::value<size_t>(), "The number of events to process. If not given, all " "available events will be processed.")
                    ("skip", boost::program_options::value<size_t>()->default_value(0), "The number of events to skip")
                    ("jobs,j", boost::program_options::value<int>()->default_value(-1), "Number of parallel jobs, negative for automatic.");
}


//---------------------------------------------------------------------------------------
void common_options::add_input_options (boost::program_options::options_description& opt)
//---------------------------------------------------------------------------------------
{
  // Add specific options for this example
  opt.add_options() ("input-dir", boost::program_options::value<std::string>()->default_value(""), "Input directory location.")
                    ("input-filename-pattern", boost::program_options::value<std::string>()->default_value(""), "Input filename pattern (common to all files).")
                    ("grid-input-dir", boost::program_options::value<std::string>()->default_value(""), "Input directory location on the grid.")
                    ("input-files", boost::program_options::value<std::vector<std::string>>(), "Input files, can occur multiple times.")
                    ("input-root", boost::program_options::value<bool>()->default_value(false), "Switch on to read '.root' file(s).")
                    ("input-csv", boost::program_options::value<bool>()->default_value(false), "Switch on to read '.csv' file(s).")
                    ("input-obj", boost::program_options::value<bool>()->default_value(false), "Switch on to read '.obj' file(s).")
                    ("input-json", boost::program_options::value<bool>()->default_value(false), "Switch on to read '.json' file(s).")
                    ("log", boost::program_options::value<std::string>()->default_value(""), "Log directory location.")
                    ("restart-from", boost::program_options::value<int>()->default_value(0), "Restart program execution from given step.")
                    ("end-at", boost::program_options::value<int>()->default_value(0), "Stop program execution at a given step.");
}


//--------------------------------------------------------------------------------------------------------------------------
void common_options::add_output_options (boost::program_options::options_description& opt, const output_format& formatFlags)
//--------------------------------------------------------------------------------------------------------------------------
{
  // Add specific options for this example
  opt.add_options() ("output-dir", boost::program_options::value<std::string>()->default_value(""), "Output directory location.");

  switch (formatFlags)
  { 
    case output_format::Root:
    opt.add_options() ("output-root", boost::program_options::bool_switch(), "Switch on to write '.root' output file(s).");
    break;
 
    case output_format::Csv:
     opt.add_options() ("output-csv", boost::program_options::bool_switch(), "Switch on to write '.csv' output file(s).");
    break;

    case output_format::Obj:
      opt.add_options() ("output-obj", boost::program_options::bool_switch(), "Switch on to write '.obj' ouput file(s).");
    break;

    case output_format::Json:
      opt.add_options() ("output-json", boost::program_options::bool_switch(), "Switch on to write '.json' ouput file(s).");
    break;

    case output_format::Txt:
      opt.add_options() ("output-txt", boost::program_options::bool_switch(), "Switch on to write '.txt' ouput file(s).");
      break;

    default:
      std::cout << "Unknown file format" << std::endl;
      exit(0);
    break;
   }
}
/*
  if (formatFlags == output_format::Root)
    opt.add_options() ("output-root", bool_switch(), "Switch on to write '.root' output file(s).");

  if (formatFlags & output_format::Csv) == output_format::Csv)
    opt.add_options() ("output-csv", bool_switch(), "Switch on to write '.csv' output file(s).");

  if (formatFlags & OutputFormat::Obj) == OutputFormat::Obj)
    opt.add_options() ("output-obj", bool_switch(), "Switch on to write '.obj' ouput file(s).");

  if (formatFlags & OutputFormat::Json) == OutputFormat::Json)
    opt.add_options() ("output-json", bool_switch(), "Switch on to write '.json' ouput file(s).");

  if (formatFlags & OutputFormat::Txt) == OutputFormat::Txt)
    opt.add_options() ("output-txt", bool_switch(), "Switch on to write '.txt' ouput file(s).");
*/


//------------------------------------------------------------------------------------------------
void common_options::add_random_numbers_options (boost::program_options::options_description& opt)
//------------------------------------------------------------------------------------------------
{
  opt.add_options() ("rnd-seed", boost::program_options::value<uint64_t>()->default_value(1234567890u), "Random numbers seed.");
}


//------------------------------------------------------------------------------------------
void common_options::add_geometry_options (boost::program_options::options_description& opt)
//------------------------------------------------------------------------------------------
{
  opt.add_options() ("geo-surface-loglevel", boost::program_options::value<size_t>()->default_value(3), "The outoput log level for the surface building.")
                    ("geo-layer-loglevel", boost::program_options::value<size_t>()->default_value(3), "The output log level for the layer building.")
                    ("geo-volume-loglevel", boost::program_options::value<size_t>()->default_value(3), "The output log level for the volume building.");
}

//------------------------------------------------------------------------------------------
void common_options::add_material_options (boost::program_options::options_description& opt)
//------------------------------------------------------------------------------------------
{
  opt.add_options() ("mat-input-type", boost::program_options::value<std::string>()->default_value("build"), "The way material is loaded: 'none', 'build', 'proto', 'file'.")
                    ("mat-input-file", boost::program_options::value<std::string>()->default_value(""), "Name of the material map input file, supported: '.json' or '.root'.")
                    ("mat-output-file", boost::program_options::value<std::string>()->default_value(""), "Name of the material map output file (without extension).")
                    ("mat-output-sensitives", boost::program_options::value<bool>()->default_value(true), "Write material information of sensitive surfaces.")
                    ("mat-output-approaches", boost::program_options::value<bool>()->default_value(true), "Write material information of approach surfaces.")
                    ("mat-output-representing", boost::program_options::value<bool>()->default_value(true), "Write material information of representing surfaces.")
                    ("mat-output-boundaries", boost::program_options::value<bool>()->default_value(true), "Write material information of boundary surfaces.")
                    ("mat-output-volumes", boost::program_options::value<bool>()->default_value(true), "Write material information of volumes.")
                    ("mat-output-dense-volumes", boost::program_options::value<bool>()->default_value(false), "Write material information of dense volumes.")
                    ("mat-output-allmaterial", boost::program_options::value<bool>()->default_value(false), "Add protoMaterial to all surfaces and volume for the mapping.");
}


//---------------------------------------------------------------------------------------
template <class T>
void common_options::add_graph_options (boost::program_options::options_description& opt)
//---------------------------------------------------------------------------------------
{
  opt.add_options () ("input-graph", boost::program_options::value<std::string>(), "Directory of the predicted graphs in graphML .txt format");
  opt.add_options () ("min-pt-cut", boost::program_options::value<T>()->default_value(0.), "Min cut on the generated particles");
  opt.add_options () ("max-pt-cut", boost::program_options::value<T>()->default_value(0.), "Max cut on the generated particles");
  opt.add_options () ("min-nhits", boost::program_options::value<long unsigned int>()->default_value(2), "Min cut of number of hits of generated particles");
  opt.add_options () ("extra-features", boost::program_options::value<bool>()->default_value(false), "Reads extra data in hit files to add edge features");
  opt.add_options () ("strip-hit-pair", boost::program_options::value<bool>()->default_value(true), "Recompute position of barrel-strip hits at edge level, for edge features (enable extra-features option)");
  opt.add_options () ("input-strip-modules", boost::program_options::value<std::string>(), " File containing strip modules DB");
  opt.add_options () ("root-filename", boost::program_options::value<std::string>(), "Name of the root file");
  opt.add_options () ("give-cut-values", boost::program_options::value<bool>()->default_value(false), "Give cuts values");
  opt.add_options () ("input-module-map", boost::program_options::value<std::string>()->default_value(""), "Name of the input module map root file");
  opt.add_options () ("output-module-map", boost::program_options::value<std::string>()->default_value(""), "Name of the output module map root file");
  opt.add_options () ("save-partial-module-maps-on-disk", boost::program_options::value<bool>()->default_value(false), "Save partial module maps for multi-threads computation");
  opt.add_options () ("output-df-name", boost::program_options::value<std::string>(), "Name of the dataframe containing a module map");
  opt.add_options () ("dump-MM-in-df", boost::program_options::value<bool>(), "Name of the dataframe containing a module map");
  opt.add_options () ("control-plot-MM", boost::program_options::value<bool>(), "Do control plot on the Module Map");
  opt.add_options () ("study-cuts-MM", boost::program_options::value<bool>(), "Study the Module Map cuts");
  opt.add_options () ("name-info-geometry", boost::program_options::value<std::string>(), "Name of geometry file with Geo ID information");
  opt.add_options () ("give-true-graph", boost::program_options::value<bool>()->default_value(false), "Give true graph");
  opt.add_options () ("save-graph-on-disk-graphml", boost::program_options::value<bool>()->default_value(false), "Save graph on disk in graphml format");
  opt.add_options () ("save-graph-on-disk-npz", boost::program_options::value<bool>()->default_value(false), "Save graph on disk in npz format");
  opt.add_options () ("save-graph-on-disk-pyg", boost::program_options::value<bool>()->default_value(false), "Save graph on disk in pyg format");
  opt.add_options () ("save-graph-on-disk-csv", boost::program_options::value<bool>()->default_value(false), "Save graph on disk in csv format");
  opt.add_options () ("MapOrGraph", boost::program_options::value<std::string>()->default_value("ModuleMap"), "Choose which script to run");
  opt.add_options () ("output-root-name", boost::program_options::value<std::string>()->default_value("efficiency.root"), "Name of the root file containing the MM efficiency");
  opt.add_options () ("input-predictedTrack-dir", boost::program_options::value<std::string>(), "Name of the directory containing the predicted tracks");
  opt.add_options () ("save-modlinks-tree", boost::program_options::value<bool>()->default_value(false), "Save module links TTree in Module Map ouput file");
  opt.add_options () ("phi-slice", boost::program_options::value<bool>()->default_value(false), "Create a slice in phi");
  opt.add_options () ("cut1-phi-slice", boost::program_options::value<T>()->default_value(0.), "Lower value of the phi cut");
  opt.add_options () ("cut2-phi-slice", boost::program_options::value<T>()->default_value(0.), "Upper value of the phi cut");
  opt.add_options () ("eta-region", boost::program_options::value<bool>()->default_value(false), "Create a reduced graph by appliying an eta cut");
  opt.add_options () ("cut1-eta", boost::program_options::value<float>()->default_value(0.), "Lower value of the eta cut");
  opt.add_options () ("cut2-eta", boost::program_options::value<float>()->default_value(0.), "Upper value of the eta cut");
  opt.add_options () ("not-follow-electron", boost::program_options::value<bool>()->default_value(false), "boolean, whether or not to follow electrons");
  opt.add_options () ("only-this-pdgID", boost::program_options::value<int>()->default_value(-1), "If value != 1, only compute efficiency for this kind of pdgID particle");
  opt.add_options () ("isML", boost::program_options::value<bool>(), "Metric Learning used to create the graphs");
  opt.add_options () ("isMM", boost::program_options::value<bool>(), "Module Map used to create the graphs");
  opt.add_options () ("maxEventId", boost::program_options::value<int>(), "Maximun event id to consider");
  opt.add_options () ("minEventId", boost::program_options::value<int>(), "Minimum event id to consider");
}


//----------------------------------------------------------------------------------------------------------------------------------------------------------
boost::program_options::variables_map common_options::parse (const boost::program_options::options_description& opt, int argc, char* argv[]) noexcept(false)
//----------------------------------------------------------------------------------------------------------------------------------------------------------
{
  boost::program_options::variables_map vm;
  boost::program_options::store (boost::program_options::command_line_parser (argc, argv).options(opt).run(), vm);
  boost::program_options::notify (vm);

  if (vm.count ("response-file") and not vm["response-file"].template as<std::string>().empty())
    { // Load the file and tokenize it
      std::ifstream ifs (vm["response-file"].as<std::string>().c_str());
      if (!ifs)
        throw (std::system_error (std::error_code(), "Could not open response file."));

      // Read the whole file into a string
      std::stringstream ss;
      ss << ifs.rdbuf();
      std::string rString = ss.str();
      std::vector<std::string> args;
      const std::regex rgx("[ \t\r\n\f]");
      std::sregex_token_iterator iter (rString.begin(), rString.end(), rgx, -1);
      std::sregex_token_iterator end;


      for (; iter != end; ++iter)
        if (!std::string(*iter).empty())
          args.push_back (*iter);

      // Parse the file and store the options
      store (boost::program_options::command_line_parser(args).options(opt).run(), vm);
    }

  // Automatically handle help
  if (vm.count("help"))
    { std::cout << opt << std::endl;
      vm.clear();
    }

  return vm;
}


/*
Acts::Logging::Level common_options::read_log_level (const boost::program_options::variables_map& vm)
{
  return Acts::Logging::Level (vm["loglevel"].as<size_t>());
}


ActsExamples::Sequencer::Config common_options::read_sequencer_config (const boost::program_options::variables_map& vm)
  {
    ActsExamples::Sequencer::Config cfg;
    cfg.skip = vm["skip"].as<size_t>();
    if (not vm["events"].empty())
      cfg.events = vm["events"].as<size_t>();
    cfg.logLevel = read_log_level (vm);
    cfg.numThreads = vm["jobs"].as<int>();
    if (!vm["output-dir"].empty())
      cfg.outputDir = vm["output-dir"].as<std::string>();

  return cfg;
}
*/


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

/*
void ActsExamples::Options::addOutputOptions(
    boost::program_options::options_description& opt,
    OutputFormat formatFlags) {
  // Add specific options for this example
  cout << "format flag = " << ACTS_CHECK_BIT(formatFlags, OutputFormat::Root) << endl;

  opt.add_options()("output-dir", value<std::string>()->default_value(""),
                    "Output directory location.");

  if (ACTS_CHECK_BIT(formatFlags, OutputFormat::Root))
    { cout << "root file format" << endl;
      opt.add_options()("output-root", bool_switch(),
                      "Switch on to write '.root' output file(s).");
    } 

  if (ACTS_CHECK_BIT(formatFlags, OutputFormat::Csv))
    { cout << "CSV file format" << endl;
      opt.add_options()("output-csv", bool_switch(),
                      "Switch on to write '.csv' output file(s).");
    }

  if (ACTS_CHECK_BIT(formatFlags, OutputFormat::Obj))
    { cout << "Obj file format" << endl;
      opt.add_options()("output-obj", bool_switch(),
                      "Switch on to write '.obj' ouput file(s).");
    }

  if (ACTS_CHECK_BIT(formatFlags, OutputFormat::Json))
    { cout << "Json file format" << endl;
      opt.add_options()("output-json", bool_switch(),
                      "Switch on to write '.json' ouput file(s).");
    }

  if (ACTS_CHECK_BIT(formatFlags, OutputFormat::Txt))
    { cout << "Txt file format" << endl;
      opt.add_options()("output-txt", bool_switch(),
                      "Switch on to write '.txt' ouput file(s).");
    }
}


Acts::Logging::Level ActsExamples::Options::readLogLevel(
    const boost::program_options::variables_map& vm) {
  return Acts::Logging::Level(vm["loglevel"].as<size_t>());
}

ActsExamples::Sequencer::Config ActsExamples::Options::readSequencerConfig(
    const boost::program_options::variables_map& vm) {
  Sequencer::Config cfg;
  cfg.skip = vm["skip"].as<size_t>();
  if (not vm["events"].empty()) {
    cfg.events = vm["events"].as<size_t>();
  }
  cfg.logLevel = readLogLevel(vm);
  cfg.numThreads = vm["jobs"].as<int>();
  if (not vm["output-dir"].empty()) {
    cfg.outputDir = vm["output-dir"].as<std::string>();
  }
  return cfg;
}

// Read the random numbers config.
ActsExamples::RandomNumbers::Config
ActsExamples::Options::readRandomNumbersConfig(
    const boost::program_options::variables_map& vm) {
  ActsExamples::RandomNumbers::Config cfg;
  cfg.seed = vm["rnd-seed"].as<uint64_t>();
  return cfg;
}

*/
