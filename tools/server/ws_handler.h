#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <functional>
#include <mutex>

struct omni_context;
struct omni_shared_models;
struct common_params;
class SessionManager;

namespace httplib {
namespace ws { class WebSocket; }
}

// ============================================================================
// WS /backend handler — main entry point called from server-omni.cpp
// ============================================================================

void handle_ws_backend(httplib::ws::WebSocket & ws,
                       SessionManager & session_mgr,
                       common_params & params_base,
                       omni_shared_models * shared);  // shared models, loaded once at startup

// ============================================================================
// Helpers: base64 audio/JPEG → temp files
// ============================================================================

struct TempMediaFiles {
    std::string audio_path;      // WAV file path (empty if no audio)
    std::string image_path;      // PNG/JPEG file path (empty if no image)
    
    // Write base64 float32 PCM to a temp WAV file
    // Returns empty string on failure
    static std::string write_audio_wav(const std::string & b64, const std::string & temp_dir, int counter);
    
    // Write base64 JPEG/PNG bytes to a temp file
    // Returns empty string on failure
    static std::string write_image_jpeg(const std::string & b64, const std::string & temp_dir, int counter);
    
    // Create a temp file from raw bytes
    static std::string write_temp_file(const std::string & temp_dir, const std::string & prefix,
                                       const std::string & suffix, const void * data, size_t len);
    
    // Clean up temp files
    void cleanup();
};
