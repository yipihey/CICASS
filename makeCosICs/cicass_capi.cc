/*********************************************************************
cicass_capi.cc -- C-ABI shim for the EnzoNG CICASS wrapper (CICASSLib).

Exposes the CICASS makeCosICs realizer (genICs) as a callable library
entry so a Julia-driven session can generate streaming-velocity ICs
in-process (or in a CodeBridge worker), then read the HDF5-free
`.cicass` raw dump written by capi_out.c.

Built into libcicass_capi.dylib by deps/build_cicass_darwin.sh with
-DCICASS_LIB -DOUTPUT_CAPI (no -DOUTPUT_GADGET/ENZO, no HDF5).

Entry points (mirrors MusicLib's music_capi surface):
  int         cicass_generate(const char *args)  // space-separated genICs flags
  const char *cicass_last_error(void)
  const char *cicass_version(void)
  int         cicass_real_bytes(void)            // sizeof field element (8 = f64)
*********************************************************************/

#include <string>
#include <vector>
#include <cstring>
#include <cstdio>

extern "C" int cicass_run_genics(int argc, char *argv[]);

static std::string g_last_error;

static std::vector<std::string> tokenize(const char *s)
{
  std::vector<std::string> out;
  if (s == nullptr) return out;
  std::string cur;
  for (const char *p = s; *p; ++p) {
    if (*p == ' ' || *p == '\t' || *p == '\n') {
      if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    } else {
      cur.push_back(*p);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

extern "C" int cicass_generate(const char *args)
{
  g_last_error.clear();
  std::vector<std::string> toks = tokenize(args);
  std::vector<char *> argv;
  std::string prog = "genICs";
  argv.push_back(const_cast<char *>(prog.c_str()));
  for (auto &t : toks) argv.push_back(const_cast<char *>(t.c_str()));

  int rc = cicass_run_genics((int)argv.size(), argv.data());
  if (rc != 0) {
    char buf[128];
    snprintf(buf, sizeof(buf), "cicass_run_genics returned %d", rc);
    g_last_error = buf;
  }
  return rc;
}

extern "C" const char *cicass_last_error(void)
{
  return g_last_error.c_str();
}

extern "C" const char *cicass_version(void)
{
  return "CICASS-EnzoNG capi 0.1 (makeCosICs/genICs)";
}

extern "C" int cicass_real_bytes(void)
{
  return (int)sizeof(double);   /* capi_out.c writes f64 fields */
}
