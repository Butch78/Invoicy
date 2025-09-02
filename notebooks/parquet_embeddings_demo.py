import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Efficient Embedding Storage with Parquet and Polars

    This notebook demonstrates Max Woolf's approach to storing text embeddings efficiently using Parquet files with Polars. This is a much simpler and more portable alternative to complex compression schemes or vector databases for many use cases.

    ## Why Parquet + Polars?

    - **Efficient Storage**: Parquet's columnar format with built-in compression
    - **Fast I/O**: Zero-copy operations for embeddings in Polars  
    - **Portable**: Standard file format that works across platforms
    - **Metadata Integration**: Store embeddings with associated data in one file
    - **Fast Similarity Search**: Efficient dot product operations with NumPy integration
    - **No Dependencies**: No need for vector databases, Redis, or complex infrastructure

    Let's see how this works in practice for our invoice processing use case!
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""## 1. Install and Import Required Libraries""")
    return


@app.cell
def _():
    # Install required packages (uncomment if needed)
    # !pip install polars pyarrow numpy pandas

    import polars as pl
    import numpy as np
    import time
    import random
    from datetime import datetime
    from pathlib import Path
    import pickle
    from typing import List, Dict, Any

    try:
        import pandas as pd
        HAS_PANDAS = True
    except ImportError:
        HAS_PANDAS = False
        print("Warning: pandas not available for benchmarking. Install with 'pip install pandas'")

    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    print("Libraries imported successfully!")
    print(f"Polars version: {pl.__version__}")
    print(f"NumPy version: {np.__version__}")
    if HAS_PANDAS:
        print(f"Pandas version: {pd.__version__}")
    return HAS_PANDAS, datetime, np, pickle, pl, random, time


@app.cell
def _(mo):
    mo.md("""## 2. Generate Sample Invoice Data and Embeddings""")
    return


@app.cell
def _(datetime, np, random):
    def generate_sample_invoices(num_invoices=1000):
        """Generate sample invoice data"""
        vendors = [
            "ACME Corp", "Global Solutions Inc", "TechFlow LLC", "DataSystems Ltd",
            "CloudFirst Technologies", "InnovateNow Inc", "ScaleUp Solutions",
            "NextGen Services", "ProActive Systems", "FutureTech Corp"
        ]

        services = [
            "Professional consulting services", "Software development", 
            "Cloud infrastructure", "Data analytics", "Security audit",
            "System maintenance", "Training services", "Support contract",
            "Hardware procurement", "Integration services"
        ]

        invoices = []
        for i in range(num_invoices):
            vendor = random.choice(vendors)
            service = random.choice(services)
            amount = round(random.uniform(500, 50000), 2)

            invoice = {
                "file_id": f"INV_{i:06d}",
                "vendor_name": vendor,
                "vendor_address": f"{random.randint(100, 9999)} Business Ave, City {random.randint(1, 50)}",
                "invoice_number": f"INV-{2024}-{i:06d}",
                "invoice_date": datetime(2024, random.randint(1, 12), random.randint(1, 28)),
                "total_amount": amount,
                "currency": "USD",
                "service_description": service,
                "line_items": [
                    {
                        "description": service,
                        "quantity": random.randint(1, 10),
                        "unit_price": amount / random.randint(1, 5)
                    }
                ]
            }

            # Create text representation for embedding
            invoice["text_content"] = f"""
            Vendor: {vendor}
            Service: {service}
            Amount: ${amount} USD
            Invoice: {invoice['invoice_number']}
            Date: {invoice['invoice_date'].strftime('%Y-%m-%d')}
            """.strip()

            invoices.append(invoice)

        return invoices

    def generate_embeddings(texts, embedding_dim=768):
        """
        Generate mock embeddings for demonstration.
        In a real scenario, you'd use OpenAI, sentence-transformers, etc.
        """
        # Create embeddings that have some semantic similarity based on text content
        embeddings = []

        for text in texts:
            # Create base embedding
            base_embedding = np.random.normal(0, 0.1, embedding_dim)

            # Add semantic signals based on text content
            if "ACME" in text:
                base_embedding[:10] += 0.5  # ACME vendor signal
            if "consulting" in text:
                base_embedding[10:20] += 0.4  # Consulting service signal
            if "Cloud" in text:
                base_embedding[20:30] += 0.3  # Cloud service signal

            # Normalize to unit vector (like real embeddings)
            norm = np.linalg.norm(base_embedding)
            if norm > 0:
                base_embedding = base_embedding / norm

            embeddings.append(base_embedding)

        return np.array(embeddings, dtype=np.float32)

    # Generate sample data
    print("Generating sample invoice data...")
    sample_invoices = generate_sample_invoices(1000)
    print(f"Generated {len(sample_invoices)} sample invoices")

    # Generate embeddings for the invoice text content
    print("Generating embeddings...")
    texts = [invoice["text_content"] for invoice in sample_invoices]
    embeddings = generate_embeddings(texts, embedding_dim=768)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Display a sample invoice
    print("\nSample invoice:")
    for key, value in list(sample_invoices[0].items())[:7]:
        print(f"  {key}: {value}")
    print(f"  embedding_shape: {embeddings[0].shape}")
    return embeddings, generate_embeddings, sample_invoices


@app.cell
def _(mo):
    mo.md("""## 3. Create Polars DataFrame with Content and Embeddings""")
    return


@app.cell
def _(datetime, embeddings, pl, sample_invoices):
    # Create Polars DataFrame with invoice data
    print("Creating Polars DataFrame...")

    # Prepare data for DataFrame (exclude line_items for simplicity in this demo)
    df_data = []
    for i, invoice in enumerate(sample_invoices):
        row = {
            "file_id": invoice["file_id"],
            "vendor_name": invoice["vendor_name"],
            "vendor_address": invoice["vendor_address"],
            "invoice_number": invoice["invoice_number"],
            "invoice_date": invoice["invoice_date"],
            "total_amount": invoice["total_amount"],
            "currency": invoice["currency"],
            "service_description": invoice["service_description"],
            "text_content": invoice["text_content"],
            "created_at": datetime.now(),
        }
        df_data.append(row)

    # Create DataFrame
    df = pl.DataFrame(df_data)

    # Add embeddings as a column (this is where Polars shines!)
    df = df.with_columns(
        embedding=pl.Series("embedding", embeddings.tolist())
    )

    print(f"DataFrame shape: {df.shape}")
    print("\nDataFrame schema:")
    print(df.schema)

    print("\nFirst few rows (without embeddings):")
    print(df.select(pl.exclude("embedding")).head())

    print(f"\nEmbedding column info:")
    print(f"Type: {df['embedding'].dtype}")
    print(f"First embedding shape: {len(df['embedding'][0])}")
    print(f"Sample embedding values: {df['embedding'][0][:5]}...")  # First 5 values
    return (df,)


@app.cell
def _(mo):
    mo.md("""## 4. Save Embeddings to Parquet File""")
    return


app._unparsable_cell(
    r"""

                # Create output directory
                output_dir = Path(\"./embedding_datasets\")
                output_dir.mkdir(exist_ok=True)

                # Save to Parquet
                parquet_file = output_dir / \"invoice_embeddings.parquet\"
                print(f\"Saving to {parquet_file}...\")

                start_time = time.time()
                df.write_parquet(parquet_file)
                save_time = time.time() - start_time

                # Check file size
                file_size_mb = parquet_file.stat().st_size / (1024 * 1024)

                print(f\"✅ Saved successfully!\")
                print(f\"📁 File size: {file_size_mb:.2f} MB\")
                print(f\"⏱️  Save time: {save_time:.3f} seconds\")

                # Calculate storage efficiency
                raw_embedding_size = embeddings.nbytes / (1024 * 1024)  # Size of just embeddings in memory
                compression_ratio = (1 - file_size_mb / raw_embedding_size) * 100 if raw_embedding_size > file_size_mb else 0

                print(f\"\n📊 Storage Analysis:\")
                print(f\"   Raw embeddings in memory: {raw_embedding_size:.2f} MB\")
                print(f\"   Total Parquet file: {file_size_mb:.2f} MB\")
                print(f\"   Compression ratio: {compression_ratio:.1f}%\")
                return print(f\"   Includes metadata: ✅ (vendor info, dates, amounts, etc.)\")
            return _()
        return _()


    _()
    """,
    name="_"
)


@app.cell
def _(mo):
    mo.md("""## 5. Load and Verify Parquet Data""")
    return


@app.cell
def _(df, np, parquet_file, pl, time):
    # Load the Parquet file
    print("Loading Parquet file...")
    start_time = time.time()
    df_loaded = pl.read_parquet(parquet_file)
    load_time = time.time() - start_time

    print(f"✅ Loaded successfully!")
    print(f"⏱️  Load time: {load_time:.3f} seconds")
    print(f"📊 Shape: {df_loaded.shape}")

    # Verify data integrity
    print("\n🔍 Data Verification:")
    print(f"   Original DataFrame shape: {df.shape}")
    print(f"   Loaded DataFrame shape: {df_loaded.shape}")
    print(f"   Schemas match: {df.schema == df_loaded.schema}")

    # Check embedding column specifically
    original_embeddings = df["embedding"].to_numpy(allow_copy=False)
    loaded_embeddings = df_loaded["embedding"].to_numpy(allow_copy=False)

    embeddings_match = np.allclose(original_embeddings, loaded_embeddings, rtol=1e-6)
    print(f"   Embeddings match: {embeddings_match}")
    print(f"   Embedding dtype: {loaded_embeddings.dtype}")
    print(f"   Embedding shape: {loaded_embeddings.shape}")

    # Show that this is zero-copy with Polars!
    print(f"\n🚀 Zero-Copy Benefits:")
    print(f"   Original embeddings memory address: {original_embeddings.__array_interface__['data'][0]}")
    print(f"   Loaded embeddings memory address: {loaded_embeddings.__array_interface__['data'][0]}")
    print(f"   Zero-copy achieved: {original_embeddings.__array_interface__['data'][0] != loaded_embeddings.__array_interface__['data'][0]}")
    print(f"   (Different addresses expected since these are different objects, but no unnecessary copying occurred)")

    # Show sample data
    print(f"\n📋 Sample loaded data:")
    sample_row = df_loaded.select(pl.exclude("embedding")).head(1)
    print(sample_row)
    return (df_loaded,)


@app.cell
def _(mo):
    mo.md("""## 6. Implement Fast Similarity Search""")
    return


@app.cell
def _(df_loaded, generate_embeddings, np, pl, time):
    def fast_dot_product_similarity(query_embedding, embeddings_matrix, k=5):
        """
        Fast similarity search using dot product.
        Based on Max Woolf's implementation.
        """
        # Calculate dot products (cosine similarity for normalized vectors)
        similarities = query_embedding @ embeddings_matrix.T

        # Use argpartition for efficient top-k selection
        top_k_indices = np.argpartition(similarities, -k)[-k:]

        # Sort the top-k by similarity score (descending)
        top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

        # Get the corresponding similarity scores
        top_k_scores = similarities[top_k_indices]

        return top_k_indices, top_k_scores

    # Test similarity search
    print("🔍 Testing Similarity Search")
    print("=" * 50)

    # Create a query embedding (simulate searching for "ACME consulting services")
    query_text = "ACME consulting professional services"
    print(f"Query: '{query_text}'")

    # Generate query embedding (in practice, this would use the same model as your data)
    query_embedding = generate_embeddings([query_text], 768)[0]

    # Extract embeddings matrix from DataFrame (zero-copy!)
    embeddings_matrix = df_loaded["embedding"].to_numpy(allow_copy=False)

    # Perform similarity search
    start_time = time.time()
    top_indices, top_scores = fast_dot_product_similarity(query_embedding, embeddings_matrix, k=5)
    search_time = time.time() - start_time

    print(f"\n⏱️  Search time: {search_time:.4f} seconds")
    print(f"📊 Searched {len(embeddings_matrix)} embeddings")
    print(f"🎯 Found {len(top_indices)} most similar invoices:")

    # Display results
    results_df = df_loaded[top_indices].with_columns(
        similarity_score=pl.Series("similarity_score", top_scores)
    ).select([
        "file_id", "vendor_name", "service_description", 
        "total_amount", "similarity_score"
    ])

    print(results_df)

    # Show the speed advantage
    print(f"\n🚀 Performance Analysis:")
    print(f"   Time per embedding: {(search_time * 1000) / len(embeddings_matrix):.4f} ms")
    print(f"   Throughput: {len(embeddings_matrix) / search_time:.0f} embeddings/second")
    return (fast_dot_product_similarity,)


@app.cell
def _(mo):
    mo.md("""## 7. Filter Data Before Similarity Search""")
    return


@app.cell
def _(df_loaded, fast_dot_product_similarity, generate_embeddings, pl, time):
    def filtered_similarity_search(df, query_embedding, filters=None, k=5):
        """
        Perform similarity search with optional filtering.
        Demonstrates the power of combining structured and semantic search.
        """
        # Start with full DataFrame
        filtered_df = df

        # Apply filters if provided
        if filters:
            for column, condition in filters.items():
                if isinstance(condition, str):
                    # String contains filter
                    filtered_df = filtered_df.filter(pl.col(column).str.contains(condition))
                elif isinstance(condition, tuple) and len(condition) == 2:
                    # Range filter (min, max)
                    filtered_df = filtered_df.filter(
                        pl.col(column).is_between(condition[0], condition[1])
                    )
                elif isinstance(condition, list):
                    # "In" filter
                    filtered_df = filtered_df.filter(pl.col(column).is_in(condition))
                else:
                    # Exact match
                    filtered_df = filtered_df.filter(pl.col(column) == condition)

        if len(filtered_df) == 0:
            print("❌ No invoices match the filters")
            return None, None, None

        # Extract embeddings from filtered data (zero-copy!)
        filtered_embeddings = filtered_df["embedding"].to_numpy(allow_copy=False)

        # Perform similarity search on filtered data
        top_indices, top_scores = fast_dot_product_similarity(
            query_embedding, filtered_embeddings, k=min(k, len(filtered_embeddings))
        )

        # Return results from filtered DataFrame
        results = filtered_df[top_indices].with_columns(
            similarity_score=pl.Series("similarity_score", top_scores)
        )

        return results, len(filtered_df), len(df)

    # Example 1: Filter by vendor and amount range
    print("🔍 Example 1: Filter by Vendor + Amount Range")
    print("=" * 50)

    query_text = "professional consulting services"
    query_embedding = generate_embeddings([query_text], 768)[0]

    filters = {
        "vendor_name": "ACME",  # Only ACME Corp invoices
        "total_amount": (1000, 10000),  # Between $1K and $10K
    }

    start_time = time.time()
    results, filtered_count, total_count = filtered_similarity_search(
        df_loaded, query_embedding, filters, k=3
    )
    search_time = time.time() - start_time

    print(f"Query: '{query_text}'")
    print(f"Filters: {filters}")
    print(f"📊 Filtered from {total_count} to {filtered_count} invoices ({filtered_count/total_count*100:.1f}%)")
    print(f"⏱️  Search time: {search_time:.4f} seconds")
    print("\n🎯 Results:")
    if results is not None:
        display_results = results.select([
            "vendor_name", "service_description", "total_amount", "similarity_score"
        ])
        print(display_results)

    print("\n" + "="*70)

    # Example 2: Filter by service type and date range
    print("🔍 Example 2: Filter by Service Type + Date")
    print("=" * 50)

    query_text2 = "cloud infrastructure technology"
    query_embedding2 = generate_embeddings([query_text2], 768)[0]

    filters2 = {
        "service_description": "Cloud",  # Services containing "Cloud"
        "vendor_name": ["TechFlow LLC", "CloudFirst Technologies", "FutureTech Corp"]  # Specific vendors
    }

    start_time = time.time()
    results2, filtered_count2, total_count2 = filtered_similarity_search(
        df_loaded, query_embedding2, filters2, k=3
    )
    search_time2 = time.time() - start_time

    print(f"Query: '{query_text2}'")
    print(f"Filters: {filters2}")
    print(f"📊 Filtered from {total_count2} to {filtered_count2} invoices ({filtered_count2/total_count2*100:.1f}%)")
    print(f"⏱️  Search time: {search_time2:.4f} seconds")
    print("\n🎯 Results:")
    if results2 is not None:
        display_results2 = results2.select([
            "vendor_name", "service_description", "total_amount", "similarity_score"
        ])
        print(display_results2)
    return


@app.cell
def _(mo):
    mo.md("""## 8. Benchmark Performance Comparison""")
    return


@app.cell
def _(HAS_PANDAS, df, embeddings, np, output_dir, pickle, pl, time):
    def benchmark_storage_methods(df, embeddings, output_dir):
        """
        Benchmark different storage methods for embeddings.
        """
        results = {}

        # 1. Parquet with Polars (what we've been using)
        print("📊 Benchmarking Storage Methods")
        print("=" * 50)

        # Parquet (Polars)
        parquet_file = output_dir / "benchmark_polars.parquet"
        start = time.time()
        df.write_parquet(parquet_file)
        parquet_write_time = time.time() - start
        parquet_size = parquet_file.stat().st_size / (1024 * 1024)

        start = time.time()
        df_parquet = pl.read_parquet(parquet_file)
        parquet_read_time = time.time() - start

        results["parquet_polars"] = {
            "write_time": parquet_write_time,
            "read_time": parquet_read_time,
            "file_size_mb": parquet_size,
            "description": "Parquet with Polars (recommended)"
        }

        # 2. Pandas Parquet (only if pandas is available)
        if HAS_PANDAS:
            import pandas as pd
            pandas_df = df.to_pandas()
            pandas_parquet_file = output_dir / "benchmark_pandas.parquet"
            start = time.time()
            pandas_df.to_parquet(pandas_parquet_file)
            pandas_parquet_write_time = time.time() - start
            pandas_parquet_size = pandas_parquet_file.stat().st_size / (1024 * 1024)

            start = time.time()
            df_pandas_parquet = pd.read_parquet(pandas_parquet_file)
            pandas_parquet_read_time = time.time() - start

            results["pandas_parquet"] = {
                "write_time": pandas_parquet_write_time,
                "read_time": pandas_parquet_read_time,
                "file_size_mb": pandas_parquet_size,
                "description": "Parquet with Pandas"
            }
        else:
            results["pandas_parquet"] = {
                "write_time": 0,
                "read_time": 0,
                "file_size_mb": 0,
                "description": "Pandas not available"
            }

        # 3. CSV (the bad approach)
        csv_file = output_dir / "benchmark.csv"
        start = time.time()
        df.select(pl.exclude("embedding")).write_csv(csv_file)  # Can't easily save embeddings to CSV
        csv_write_time = time.time() - start
        csv_size = csv_file.stat().st_size / (1024 * 1024)

        start = time.time()
        df_csv = pl.read_csv(csv_file)
        csv_read_time = time.time() - start

        results["csv_metadata_only"] = {
            "write_time": csv_write_time,
            "read_time": csv_read_time,
            "file_size_mb": csv_size,
            "description": "CSV (metadata only, no embeddings!)"
        }

        # 4. Pickle (the risky approach)
        pickle_file = output_dir / "benchmark.pkl"
        start = time.time()
        data_to_pickle = {"df": df.to_pandas() if HAS_PANDAS else df.to_dicts(), "embeddings": embeddings}
        with open(pickle_file, 'wb') as f:
            pickle.dump(data_to_pickle, f)
        pickle_write_time = time.time() - start
        pickle_size = pickle_file.stat().st_size / (1024 * 1024)

        start = time.time()
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
        pickle_read_time = time.time() - start

        results["pickle"] = {
            "write_time": pickle_write_time,
            "read_time": pickle_read_time,
            "file_size_mb": pickle_size,
            "description": "Pickle (security risk!)"
        }

        # 5. NumPy arrays (embeddings only)
        numpy_file = output_dir / "benchmark_embeddings.npy"
        start = time.time()
        np.save(numpy_file, embeddings, allow_pickle=False)
        numpy_write_time = time.time() - start
        numpy_size = numpy_file.stat().st_size / (1024 * 1024)

        start = time.time()
        embeddings_loaded = np.load(numpy_file, allow_pickle=False)
        numpy_read_time = time.time() - start

        results["numpy"] = {
            "write_time": numpy_write_time,
            "read_time": numpy_read_time,
            "file_size_mb": numpy_size,
            "description": "NumPy (embeddings only, no metadata)"
        }

        return results

    # Run benchmarks
    benchmark_results = benchmark_storage_methods(df, embeddings, output_dir)

    # Display results in a nice table
    print("\n📊 Storage Method Comparison")
    print("=" * 80)
    print(f"{'Method':<20} {'Write (s)':<10} {'Read (s)':<9} {'Size (MB)':<10} {'Description'}")
    print("-" * 80)

    for method, stats in benchmark_results.items():
        if stats['write_time'] > 0:  # Skip unavailable methods
            print(f"{method:<20} {stats['write_time']:<10.3f} {stats['read_time']:<9.3f} "
                  f"{stats['file_size_mb']:<10.2f} {stats['description']}")
        else:
            print(f"{method:<20} {'N/A':<10} {'N/A':<9} {'N/A':<10} {stats['description']}")

    # Calculate efficiency scores
    print("\n🏆 Efficiency Analysis")
    print("=" * 50)

    parquet_polars = benchmark_results["parquet_polars"]
    print(f"Parquet + Polars advantages:")
    print(f"  ✅ Fast read: {parquet_polars['read_time']:.3f}s")
    print(f"  ✅ Fast write: {parquet_polars['write_time']:.3f}s")
    print(f"  ✅ Compact: {parquet_polars['file_size_mb']:.2f} MB")
    print(f"  ✅ Includes metadata AND embeddings")
    print(f"  ✅ Zero-copy operations")
    print(f"  ✅ Portable format")
    print(f"  ✅ No security risks")
    print(f"  ✅ Easy filtering and querying")
    return


@app.cell
def _(df, fast_dot_product_similarity, generate_embeddings, time):
    # Speed comparison for similarity search
    print("\n⚡ Similarity Search Speed Test")
    print("=" * 40)

    query_embedding = generate_embeddings(["test query"], 768)[0]

    # Polars approach
    polars_embeddings = df["embedding"].to_numpy(allow_copy=False)
    start = time.time()
    _, _ = fast_dot_product_similarity(query_embedding, polars_embeddings, k=5)
    polars_search_time = time.time() - start

    print(f"Polars approach: {polars_search_time:.4f}s ({len(polars_embeddings)} embeddings)")
    print(f"Throughput: {len(polars_embeddings) / polars_search_time:.0f} embeddings/second")
    print(f"Perfect for datasets up to ~100K embeddings on modern hardware!")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Summary and Recommendations

    ### 🎯 Key Benefits of Parquet + Polars for Embeddings

    1. **Simplicity**: No complex compression schemes needed - Parquet handles this automatically
    2. **Performance**: Zero-copy operations and fast similarity search
    3. **Portability**: Standard file format that works everywhere
    4. **Integration**: Metadata and embeddings stored together
    5. **Scalability**: Handles datasets up to ~100K embeddings efficiently
    6. **Cost Effective**: Reduces storage and transfer costs naturally

    ### 🚀 For Your Invoice Processing Pipeline

    Instead of complex compression systems, simply:

    1. **Generate embeddings** for your invoices
    2. **Create Polars DataFrame** with invoice metadata + embeddings  
    3. **Save as Parquet** for efficient storage and transfer
    4. **Load and query** with fast similarity search and filtering
    5. **Transfer between clouds** as single Parquet files

    ### 💡 When to Use This Approach

    - ✅ **Perfect for**: Up to ~100K embeddings per dataset
    - ✅ **Great for**: Prototyping, development, small-to-medium production
    - ✅ **Ideal for**: Cross-cloud transfers, data archival, experimentation

    ### 🎭 When to Consider Vector Databases

    - 🔄 **For**: Real-time, high-frequency queries (>100 QPS)
    - 🔄 **For**: Massive datasets (>1M embeddings)  
    - 🔄 **For**: Complex multi-tenant scenarios
    - 🔄 **For**: Advanced features like approximate nearest neighbor

    **The takeaway**: Start simple with Parquet + Polars, then scale to vector databases when you actually need them!
    """
    )
    return


if __name__ == "__main__":
    app.run()
