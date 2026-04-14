#include "ExitHandler.h"
#include "FringeEvalRL.h"
#include <fstream>
#include <regex>

// --- Singleton instance initialization ---
template <StateRepresentation StateRepr>
FringeEvalRL<StateRepr> *FringeEvalRL<StateRepr>::instance = nullptr;

template <StateRepresentation StateRepr>
FringeEvalRL<StateRepr> &FringeEvalRL<StateRepr>::get_instance() {
    if (!instance) {
        ExitHandler::exit_with_message(
            ExitHandler::ExitCode::FringeEvalInstanceError,
            "FringeEvalRL instance not created. Call create_instance() first.");
        std::exit(static_cast<int>(ExitHandler::ExitCode::ExitForCompiler));
    }
    return *instance;
}

template <StateRepresentation StateRepr>
void FringeEvalRL<StateRepr>::create_instance() {
    if (!instance) {
        instance = new FringeEvalRL();
    }
}

template <StateRepresentation StateRepr>
FringeEvalRL<StateRepr>::FringeEvalRL() {
    
    // Create GNN if not created yet
    GraphNN<StateRepr>::create_instance();
    initialize_onnx_model();
}

template <StateRepresentation StateRepr>
void FringeEvalRL<StateRepr>::initialize_onnx_model() {
    if (m_model_loaded)
        return;

    try {
        m_session_options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL);

        /*#ifdef _WIN32
            // Windows way
            _putenv_s("ORT_CUDA_USE_CUDNN", "0");
            _putenv_s("CUDA_LAUNCH_BLOCKING", "1");  // optional, forces sync errors
        #else
            // Linux / WSL / macOS way
            setenv("ORT_CUDA_USE_CUDNN", "0", 1);
            setenv("CUDA_LAUNCH_BLOCKING", "1", 1);  // optional
        #endif*/

        // Add this line to show warnings (2) to complete verbose (0) (only errors
        // and above will be shown)
        if (ArgumentParser::get_instance().get_verbose()) {
            m_session_options.SetLogSeverityLevel(0);
        }

#ifdef USE_CUDA
        try {
            OrtCUDAProviderOptions cuda_options;
            m_session_options.AppendExecutionProvider_CUDA(cuda_options);
            if (ArgumentParser::get_instance().get_verbose()) {
                ArgumentParser::get_instance().get_output_stream()
                    << "[ONNX] CUDA execution provider enabled via USE_CUDA."
                    << std::endl;
            }
        } catch (const Ort::Exception &e) {
            ArgumentParser::get_instance().get_output_stream()
                << "[WARNING][ONNX] Failed to enable CUDA, defaulting to CPU: "
                << e.what() << std::endl;
        }
#else
        if (ArgumentParser::get_instance().get_verbose()) {
            ArgumentParser::get_instance().get_output_stream()
                << "[ONNX] Compiled without CUDA (USE_CUDA not defined), using CPU."
                << std::endl;
        }
#endif

        m_session = std::make_unique<Ort::Session>(m_env, m_model_path.c_str(),
                                                   m_session_options);
        m_allocator = std::make_unique<Ort::AllocatorWithDefaultOptions>();
        m_memory_info = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

        // Fixed: No allocator argument
        m_input_names = m_session->GetInputNames();
        m_output_names = m_session->GetOutputNames();

        m_model_loaded = true;
    } catch (const std::exception &e) {
        ExitHandler::exit_with_message(ExitHandler::ExitCode::GNNModelLoadError,
                                       std::string("Failed to load ONNX model: ") +
                                           e.what());
    }

    if (ArgumentParser::get_instance().get_verbose()) {
        auto &os = ArgumentParser::get_instance().get_output_stream()
                   << "[ONNX] Model loaded: " << m_model_path << std::endl;
        os << "[ONNX] Model Constant loaded from: "
           << Configuration::get_instance().get_GNN_constant_path() << std::endl;

        // Print model input and output details
        const auto input_names = m_session->GetInputNames();
        const auto output_names = m_session->GetOutputNames();

        os << "[ONNX] Model Inputs:\n";
        for (size_t i = 0; i < input_names.size(); ++i) {
            const auto &name = input_names[i];
            auto type_info = m_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            const auto element_type = tensor_info.GetElementType();
            auto shape = tensor_info.GetShape();

            os << "  Name: " << name << "\n";
            os << "  Type: " << element_type << "\n";
            os << "  Shape: [";
            for (size_t j = 0; j < shape.size(); ++j) {
                os << shape[j];
                if (j < shape.size() - 1)
                    os << ", ";
            }
            os << "]\n";
        }

        os << "[ONNX] Model Outputs:\n";
        for (size_t i = 0; i < output_names.size(); ++i) {
            const auto &name = output_names[i];
            auto type_info = m_session->GetOutputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            auto element_type = tensor_info.GetElementType();
            auto shape = tensor_info.GetShape();

            os << "  Name: " << name << "\n";
            os << "  Type: " << element_type << "\n";
            os << "  Shape: [";
            for (size_t j = 0; j < shape.size(); ++j) {
                os << shape[j];
                if (j < shape.size() - 1)
                    os << ", ";
            }
            os << "]\n";
        }
        os << "[ONNX] Model successfully printed." << std::endl;
    }
}


template<StateRepresentation StateRepr>
FringeTensor FringeEvalRL<StateRepr>::fringe_to_tensor_minimal(const std::vector<State<StateRepr>> &states) {
    //Here combine the various tensor in one, each state has it. Need to modify the search so that it modies a list of states but doubale

    //The fringe will have n spots, the first m will be the newly expanded and the remaining will be the top of the previous fringe.
    //Everything will be assigned a score, when something exceed the fringe will be
    // assigned its score using a different heuristics? Or maybe using the depth?

    /*Facciamo finta di avere 4 stati e fringe massima di 5 stati, espandiamo il piu' promettente che ne genera altri 4. Questo significa che teniamo i 4 stati piu' il secondo piu' promettente. Ora ripetiamo e di nuovo generiamo altri 4 stati. Questo vuol dire che nella coda fuori dalla fringe abbiamo 7 stati che non sono comparabili tra loro, cosa facciamo? Come prediamo i piu' promettenti tra quei 7 che non hanno mai fatto una frangia insime e quindi non possono essere ordinati? Abbiamo ordinamento tra alcuni subset ma non tutti. Forse ci conviene usare un'altra euristica e garantire l'ordinamento dei subset che abbiamo controllato??*/
}