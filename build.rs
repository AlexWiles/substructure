use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    // Generate prost types + file descriptor set
    let descriptor_path = out_dir.join("proto_descriptor.bin");
    prost_build::Config::new()
        .file_descriptor_set_path(&descriptor_path)
        .compile_protos(&["proto/a2a.proto"], &["proto/"])?;

    // prost_build maps google.protobuf WKTs to ::prost_types, which lack serde impls.
    // Rewrite to use ::pbjson_types (drop-in replacements with serde support).
    let prost_path = out_dir.join("a2a.rs");
    let code = std::fs::read_to_string(&prost_path)?;
    std::fs::write(&prost_path, code.replace("::prost_types::", "::pbjson_types::"))?;

    // Generate serde impls for our A2A types
    let descriptor_set = std::fs::read(&descriptor_path)?;
    pbjson_build::Builder::new()
        .register_descriptors(&descriptor_set)?
        .extern_path(".google.protobuf", "::pbjson_types")
        .build(&[".a2a"])?;

    Ok(())
}
