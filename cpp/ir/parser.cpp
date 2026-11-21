#include "ir/parser.hpp"

#include <cctype>
#include <charconv>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace m1triton::ir {

namespace {

enum class TokKind {
  Eof,
  Ident,
  ValueRef,
  SymbolRef,
  Number,
  LParen,
  RParen,
  LBrace,
  RBrace,
  LBracket,
  RBracket,
  LAngle,
  RAngle,
  Comma,
  Colon,
  Equals,
  Dot,
};

struct Token {
  TokKind kind = TokKind::Eof;
  std::string text;
  size_t line = 1;
  size_t col = 1;
};

class Lexer {
 public:
  explicit Lexer(std::string_view src) : src_(src) {}

  std::vector<Token> tokenize() {
    std::vector<Token> out;
    while (pos_ < src_.size()) {
      skip_ws_and_comments();
      if (pos_ >= src_.size()) break;
      const size_t start_line = line_;
      const size_t start_col = col_;
      const char c = src_[pos_];

      auto emit_single = [&](TokKind k) {
        out.push_back(Token{k, std::string(1, c), start_line, start_col});
        advance();
      };

      switch (c) {
        case '(':
          emit_single(TokKind::LParen);
          continue;
        case ')':
          emit_single(TokKind::RParen);
          continue;
        case '{':
          emit_single(TokKind::LBrace);
          continue;
        case '}':
          emit_single(TokKind::RBrace);
          continue;
        case '[':
          emit_single(TokKind::LBracket);
          continue;
        case ']':
          emit_single(TokKind::RBracket);
          continue;
        case '<':
          emit_single(TokKind::LAngle);
          continue;
        case '>':
          emit_single(TokKind::RAngle);
          continue;
        case ',':
          emit_single(TokKind::Comma);
          continue;
        case ':':
          emit_single(TokKind::Colon);
          continue;
        case '=':
          emit_single(TokKind::Equals);
          continue;
        case '.':
          emit_single(TokKind::Dot);
          continue;
        default:
          break;
      }

      if (c == '%' || c == '@') {
        const TokKind kind = (c == '%') ? TokKind::ValueRef : TokKind::SymbolRef;
        advance();
        const size_t begin = pos_;
        while (pos_ < src_.size() && (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) {
          advance();
        }
        if (pos_ == begin) {
          throw std::runtime_error("Empty name after '" + std::string(1, c) + "' at line " +
                                   std::to_string(start_line));
        }
        out.push_back(Token{kind, std::string(src_.substr(begin, pos_ - begin)), start_line, start_col});
        continue;
      }

      if (std::isdigit(static_cast<unsigned char>(c)) ||
          (c == '-' && pos_ + 1 < src_.size() &&
           (std::isdigit(static_cast<unsigned char>(src_[pos_ + 1])) ||
            src_.substr(pos_ + 1, 3) == "inf"))) {
        const size_t begin = pos_;
        if (c == '-') advance();
        if (src_.substr(pos_, 3) == "inf") {
          advance();
          advance();
          advance();
          out.push_back(Token{TokKind::Number, std::string(src_.substr(begin, pos_ - begin)), start_line, start_col});
          continue;
        }
        while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
          advance();
        }
        if (pos_ < src_.size() && src_[pos_] == '.') {
          advance();
          while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
            advance();
          }
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
          advance();
          if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) advance();
          while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) {
            advance();
          }
        }
        if (pos_ < src_.size() && src_[pos_] == 'u') {
          advance();
        }
        out.push_back(Token{TokKind::Number, std::string(src_.substr(begin, pos_ - begin)), start_line, start_col});
        continue;
      }

      if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
        const size_t begin = pos_;
        while (pos_ < src_.size() && (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_')) {
          advance();
        }
        out.push_back(Token{TokKind::Ident, std::string(src_.substr(begin, pos_ - begin)), start_line, start_col});
        continue;
      }

      throw std::runtime_error("Unexpected character '" + std::string(1, c) + "' at line " +
                               std::to_string(start_line) + ", col " + std::to_string(start_col));
    }
    out.push_back(Token{TokKind::Eof, "", line_, col_});
    return out;
  }

 private:
  void advance() {
    if (src_[pos_] == '\n') {
      ++line_;
      col_ = 1;
    } else {
      ++col_;
    }
    ++pos_;
  }

  void skip_ws_and_comments() {
    while (pos_ < src_.size()) {
      if (std::isspace(static_cast<unsigned char>(src_[pos_]))) {
        advance();
      } else if (pos_ + 1 < src_.size() && src_[pos_] == '/' && src_[pos_ + 1] == '/') {
        while (pos_ < src_.size() && src_[pos_] != '\n') advance();
      } else {
        break;
      }
    }
  }

  std::string_view src_;
  size_t pos_ = 0;
  size_t line_ = 1;
  size_t col_ = 1;
};

std::optional<UnaryOp> parse_unary_op(std::string_view s) {
  if (s == "exp") return UnaryOp::Exp;
  if (s == "log") return UnaryOp::Log;
  if (s == "sqrt") return UnaryOp::Sqrt;
  if (s == "rsqrt") return UnaryOp::Rsqrt;
  if (s == "abs") return UnaryOp::Abs;
  if (s == "neg") return UnaryOp::Neg;
  if (s == "tanh") return UnaryOp::Tanh;
  if (s == "erf") return UnaryOp::Erf;
  if (s == "floor") return UnaryOp::Floor;
  if (s == "ceil") return UnaryOp::Ceil;
  if (s == "sin") return UnaryOp::Sin;
  if (s == "cos") return UnaryOp::Cos;
  if (s == "sigmoid") return UnaryOp::Sigmoid;
  if (s == "not") return UnaryOp::Not;
  return std::nullopt;
}

std::optional<BinaryOp> parse_binary_op(std::string_view s) {
  if (s == "add") return BinaryOp::Add;
  if (s == "sub") return BinaryOp::Sub;
  if (s == "mul") return BinaryOp::Mul;
  if (s == "div") return BinaryOp::Div;
  if (s == "rem") return BinaryOp::Rem;
  if (s == "max") return BinaryOp::Max;
  if (s == "min") return BinaryOp::Min;
  if (s == "pow") return BinaryOp::Pow;
  if (s == "and") return BinaryOp::And;
  if (s == "or") return BinaryOp::Or;
  if (s == "xor") return BinaryOp::Xor;
  if (s == "shl") return BinaryOp::Shl;
  if (s == "shr") return BinaryOp::Shr;
  if (s == "cmp_eq") return BinaryOp::CmpEq;
  if (s == "cmp_ne") return BinaryOp::CmpNe;
  if (s == "cmp_lt") return BinaryOp::CmpLt;
  if (s == "cmp_le") return BinaryOp::CmpLe;
  if (s == "cmp_gt") return BinaryOp::CmpGt;
  if (s == "cmp_ge") return BinaryOp::CmpGe;
  return std::nullopt;
}

std::optional<OpKind> parse_op_kind(std::string_view s) {
  if (s == "const") return OpKind::Constant;
  if (s == "program_id") return OpKind::ProgramId;
  if (s == "num_programs") return OpKind::NumPrograms;
  if (s == "arange") return OpKind::Arange;
  if (s == "splat") return OpKind::Splat;
  if (s == "expand_dims") return OpKind::ExpandDims;
  if (s == "broadcast") return OpKind::Broadcast;
  if (s == "reshape") return OpKind::Reshape;
  if (s == "permute") return OpKind::Permute;
  if (s == "cast") return OpKind::Cast;
  if (s == "unary") return OpKind::Unary;
  if (s == "binary") return OpKind::Binary;
  if (s == "select") return OpKind::Select;
  if (s == "ptr_add") return OpKind::PtrAdd;
  if (s == "load") return OpKind::Load;
  if (s == "store") return OpKind::Store;
  if (s == "atomic_add") return OpKind::AtomicAdd;
  if (s == "for_loop") return OpKind::ForLoop;
  if (s == "if") return OpKind::If;
  if (s == "yield") return OpKind::Yield;
  if (s == "return") return OpKind::Return;
  if (s == "dot") return OpKind::Dot;
  if (s == "reduce") return OpKind::Reduce;
  if (s == "scan") return OpKind::Scan;
  if (s == "welford_reduce") return OpKind::WelfordReduce;
  if (s == "welford_combine") return OpKind::WelfordCombine;
  if (s == "online_softmax") return OpKind::OnlineSoftmax;
  if (s == "online_softmax_step") return OpKind::OnlineSoftmaxStep;
  if (s == "implicit_gemm_conv") return OpKind::ImplicitGemmConv;
  if (s == "implicit_gemm_load") return OpKind::ImplicitGemmLoad;
  if (s == "flash_attention_tile") return OpKind::FlashAttentionTile;
  if (s == "threadgroup_alloc") return OpKind::ThreadgroupAlloc;
  if (s == "threadgroup_stage") return OpKind::ThreadgroupStage;
  if (s == "barrier") return OpKind::Barrier;
  if (s == "simdgroup_matmul") return OpKind::SimdgroupMatmul;
  if (s == "simd_reduce") return OpKind::SimdReduce;
  if (s == "simd_scan") return OpKind::SimdScan;
  return std::nullopt;
}

AddrSpace parse_addr_space(std::string_view s) {
  if (s == "device") return AddrSpace::Device;
  if (s == "constant") return AddrSpace::Constant;
  if (s == "threadgroup") return AddrSpace::Threadgroup;
  throw std::runtime_error("Unknown address space: " + std::string(s));
}

MatrixStructure parse_matrix_structure(std::string_view s) {
  if (s == "dense") return MatrixStructure::Dense;
  if (s == "upper_triangular") return MatrixStructure::UpperTriangular;
  if (s == "lower_triangular") return MatrixStructure::LowerTriangular;
  if (s == "symmetric") return MatrixStructure::Symmetric;
  if (s == "diagonal") return MatrixStructure::Diagonal;
  if (s == "causal_mask") return MatrixStructure::CausalMask;
  throw std::runtime_error("Unknown matrix structure: " + std::string(s));
}

LayoutKind parse_layout_kind(std::string_view s) {
  if (s == "unassigned") return LayoutKind::Unassigned;
  if (s == "blocked") return LayoutKind::Blocked;
  if (s == "tg_staged") return LayoutKind::ThreadgroupStaged;
  if (s == "simd_frag") return LayoutKind::SimdgroupFragment;
  if (s == "replicated") return LayoutKind::Replicated;
  throw std::runtime_error("Unknown layout kind: " + std::string(s));
}

SwizzleMode parse_swizzle_mode(std::string_view s) {
  if (s == "none") return SwizzleMode::None;
  if (s == "pad4") return SwizzleMode::PadBank4;
  if (s == "pad8") return SwizzleMode::PadBank8;
  if (s == "xor8") return SwizzleMode::Xor8;
  throw std::runtime_error("Unknown swizzle mode: " + std::string(s));
}

FragRole parse_frag_role(std::string_view s) {
  if (s == "a") return FragRole::MatrixA;
  if (s == "b") return FragRole::MatrixB;
  if (s == "acc") return FragRole::Accumulator;
  throw std::runtime_error("Unknown frag role: " + std::string(s));
}

TensorLayoutHint parse_tensor_layout_hint(std::string_view s) {
  if (s == "nchw") return TensorLayoutHint::NCHW;
  if (s == "nhwc") return TensorLayoutHint::NHWC;
  if (s == "ncdhw") return TensorLayoutHint::NCDHW;
  if (s == "row_major") return TensorLayoutHint::RowMajor;
  if (s == "col_major") return TensorLayoutHint::ColMajor;
  throw std::runtime_error("Unknown tensor layout hint: " + std::string(s));
}

ReduceKind parse_reduce_kind(std::string_view s) {
  if (s == "sum") return ReduceKind::Sum;
  if (s == "max") return ReduceKind::Max;
  if (s == "min") return ReduceKind::Min;
  if (s == "argmax") return ReduceKind::Argmax;
  if (s == "argmin") return ReduceKind::Argmin;
  throw std::runtime_error("Unknown reduce kind: " + std::string(s));
}

ReduceResultMode parse_reduce_mode(std::string_view s) {
  if (s == "value") return ReduceResultMode::ValueOnly;
  if (s == "index") return ReduceResultMode::IndexOnly;
  if (s == "value_and_index") return ReduceResultMode::ValueAndIndex;
  throw std::runtime_error("Unknown reduce mode: " + std::string(s));
}

ScanKind parse_scan_kind(std::string_view s) {
  if (s == "cumsum") return ScanKind::Cumsum;
  if (s == "cumprod") return ScanKind::Cumprod;
  throw std::runtime_error("Unknown scan kind: " + std::string(s));
}

BarrierScope parse_barrier_scope(std::string_view s) {
  if (s == "simdgroup") return BarrierScope::Simdgroup;
  if (s == "threadgroup") return BarrierScope::Threadgroup;
  if (s == "device_mem") return BarrierScope::DeviceMem;
  throw std::runtime_error("Unknown barrier scope: " + std::string(s));
}

class Parser {
 public:
  explicit Parser(std::vector<Token> tokens) : tokens_(std::move(tokens)) {}

  std::unique_ptr<Module> parse_module() {
    auto mod = std::make_unique<Module>();
    while (!check(TokKind::Eof)) {
      parse_kernel(*mod);
    }
    return mod;
  }

 private:
  const Token& peek(size_t ahead = 0) const {
    const size_t idx = std::min(pos_ + ahead, tokens_.size() - 1);
    return tokens_[idx];
  }

  const Token& consume() {
    const Token& t = peek();
    if (pos_ + 1 < tokens_.size()) ++pos_;
    return t;
  }

  bool check(TokKind k) const { return peek().kind == k; }
  bool check_ident(std::string_view s) const { return peek().kind == TokKind::Ident && peek().text == s; }

  const Token& expect(TokKind k, const char* what) {
    if (!check(k)) {
      throw std::runtime_error("Expected " + std::string(what) + " at line " + std::to_string(peek().line) +
                               ", got '" + peek().text + "'");
    }
    return consume();
  }

  void expect_ident(std::string_view s) {
    if (!check_ident(s)) {
      throw std::runtime_error("Expected '" + std::string(s) + "' at line " + std::to_string(peek().line) +
                               ", got '" + peek().text + "'");
    }
    consume();
  }

  bool parse_bool() {
    const Token& t = expect(TokKind::Ident, "bool");
    if (t.text == "true") return true;
    if (t.text == "false") return false;
    throw std::runtime_error("Invalid bool literal '" + t.text + "' at line " + std::to_string(t.line));
  }

  int64_t parse_i64() {
    const Token& t = expect(TokKind::Number, "integer");
    return std::stoll(t.text);
  }

  uint32_t parse_u32() {
    const Token& t = expect(TokKind::Number, "unsigned integer");
    return static_cast<uint32_t>(std::stoull(t.text));
  }

  double parse_double() {
    if (check_ident("nan")) {
      consume();
      return std::numeric_limits<double>::quiet_NaN();
    }
    if (check_ident("inf")) {
      consume();
      return std::numeric_limits<double>::infinity();
    }
    const Token& t = expect(TokKind::Number, "number");
    if (t.text == "-inf") return -std::numeric_limits<double>::infinity();
    return std::stod(t.text);
  }

  // Parses shape or grid tokens like "64x16" or "16x16x1" which lex as Number, Ident("x16") or Number, Ident("x"), Number.
  std::vector<int64_t> parse_x_dims() {
    std::vector<int64_t> dims;
    if (check(TokKind::Number) && peek().text == "0" && peek(1).kind == TokKind::Ident && peek(1).text == "d") {
      consume();
      consume();
      return dims;
    }
    const Token& first = expect(TokKind::Number, "dimension");
    dims.push_back(std::stoll(first.text));
    while (check(TokKind::Ident) && !peek().text.empty() && peek().text[0] == 'x') {
      const std::string tok = consume().text;
      size_t i = 0;
      while (i < tok.size()) {
        if (tok[i] != 'x') {
          throw std::runtime_error("Malformed shape token '" + tok + "'");
        }
        ++i;
        const size_t num_start = i;
        while (i < tok.size() && std::isdigit(static_cast<unsigned char>(tok[i]))) {
          ++i;
        }
        if (i == num_start) {
          dims.push_back(parse_i64());
        } else {
          dims.push_back(std::stoll(tok.substr(num_start, i - num_start)));
        }
      }
    }
    return dims;
  }

  StructureAttr parse_structure_attr() {
    StructureAttr s;
    s.kind = parse_matrix_structure(expect(TokKind::Ident, "structure kind").text);
    if (check(TokKind::LParen)) {
      consume();
      while (!check(TokKind::RParen)) {
        const std::string k = expect(TokKind::Ident, "structure key").text;
        expect(TokKind::Equals, "'='");
        if (k == "off") {
          s.diagonal_offset = static_cast<int32_t>(parse_i64());
        } else if (k == "unit") {
          s.is_unit_diagonal = parse_bool();
        } else {
          throw std::runtime_error("Unknown structure field: " + k);
        }
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RParen, "')'");
    }
    return s;
  }

  LayoutAttr parse_layout_attr() {
    LayoutAttr l;
    l.kind = parse_layout_kind(expect(TokKind::Ident, "layout kind").text);
    if (check(TokKind::LParen)) {
      consume();
      while (!check(TokKind::RParen)) {
        const std::string k = expect(TokKind::Ident, "layout key").text;
        expect(TokKind::Equals, "'='");
        if (k == "vec") {
          l.vec_width = parse_u32();
        } else if (k == "align") {
          l.alignment_bytes = parse_u32();
        } else if (k == "contig") {
          l.contiguity_elems = parse_u32();
        } else if (k == "tg") {
          auto d = parse_x_dims();
          if (d.size() != 3) throw std::runtime_error("tg expects 3 dimensions");
          l.tg_threads[0] = static_cast<uint32_t>(d[0]);
          l.tg_threads[1] = static_cast<uint32_t>(d[1]);
          l.tg_threads[2] = static_cast<uint32_t>(d[2]);
        } else if (k == "ept") {
          auto d = parse_x_dims();
          if (d.size() != 2) throw std::runtime_error("ept expects 2 dimensions");
          l.elems_per_thread[0] = static_cast<uint32_t>(d[0]);
          l.elems_per_thread[1] = static_cast<uint32_t>(d[1]);
        } else if (k == "swizzle") {
          l.swizzle = parse_swizzle_mode(expect(TokKind::Ident, "swizzle").text);
        } else if (k == "pad") {
          l.pad_elems = parse_u32();
        } else if (k == "stages") {
          l.stages = parse_u32();
        } else if (k == "role") {
          l.frag_role = parse_frag_role(expect(TokKind::Ident, "role").text);
        } else if (k == "ftiles") {
          auto d = parse_x_dims();
          if (d.size() != 2) throw std::runtime_error("ftiles expects 2 dimensions");
          l.frag_tiles_m = static_cast<uint32_t>(d[0]);
          l.frag_tiles_n = static_cast<uint32_t>(d[1]);
        } else if (k == "sgrid") {
          auto d = parse_x_dims();
          if (d.size() != 2) throw std::runtime_error("sgrid expects 2 dimensions");
          l.simd_grid[0] = static_cast<uint32_t>(d[0]);
          l.simd_grid[1] = static_cast<uint32_t>(d[1]);
        } else {
          throw std::runtime_error("Unknown layout field: " + k);
        }
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RParen, "')'");
    }
    return l;
  }

  PtrType parse_ptr_type() {
    expect_ident("ptr");
    expect(TokKind::LAngle, "'<'");
    const std::string dt_str = expect(TokKind::Ident, "pointee dtype").text;
    auto dt = parse_dtype(dt_str);
    if (!dt) throw std::runtime_error("Unknown pointee dtype: " + dt_str);
    AddrSpace space = AddrSpace::Device;
    if (check(TokKind::Comma)) {
      consume();
      space = parse_addr_space(expect(TokKind::Ident, "addr space").text);
    }
    expect(TokKind::RAngle, "'>'");
    return PtrType{*dt, space};
  }

  Type parse_type() {
    if (check_ident("ptr")) {
      const PtrType p = parse_ptr_type();
      return Type::ptr(p.pointee, p.space);
    }
    if (check_ident("tile")) {
      consume();
      expect(TokKind::LAngle, "'<'");
      auto shape = parse_x_dims();
      expect(TokKind::Comma, "','");
      ElemType elem;
      if (check_ident("ptr")) {
        elem = parse_ptr_type();
      } else {
        const std::string dt_str = expect(TokKind::Ident, "elem dtype").text;
        auto dt = parse_dtype(dt_str);
        if (!dt) throw std::runtime_error("Unknown tile element dtype: " + dt_str);
        elem = *dt;
      }
      LayoutAttr layout;
      StructureAttr structure;
      while (check(TokKind::Comma)) {
        consume();
        const std::string key = expect(TokKind::Ident, "tile attr key").text;
        expect(TokKind::Equals, "'='");
        if (key == "layout") {
          layout = parse_layout_attr();
        } else if (key == "struct") {
          structure = parse_structure_attr();
        } else {
          throw std::runtime_error("Unknown tile attribute: " + key);
        }
      }
      expect(TokKind::RAngle, "'>'");
      return Type::tile(std::move(shape), elem, layout, structure);
    }
    const Token& t = expect(TokKind::Ident, "type");
    auto dt = parse_dtype(t.text);
    if (!dt) {
      throw std::runtime_error("Unknown type '" + t.text + "' at line " + std::to_string(t.line));
    }
    return Type::scalar(*dt);
  }

  std::vector<Type> parse_type_or_tuple() {
    std::vector<Type> out;
    if (check(TokKind::LParen)) {
      consume();
      while (!check(TokKind::RParen)) {
        out.push_back(parse_type());
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RParen, "')'");
    } else {
      out.push_back(parse_type());
    }
    return out;
  }

  OpHints parse_optional_hints() {
    OpHints hints;
    if (!check_ident("hints")) return hints;
    consume();
    expect(TokKind::LBrace, "'{'");
    while (!check(TokKind::RBrace)) {
      const std::string k = expect(TokKind::Ident, "hint key").text;
      if (k == "tile_spec") {
        TileSpecAttr ts;
        expect(TokKind::LParen, "'('");
        while (!check(TokKind::RParen)) {
          const std::string fk = expect(TokKind::Ident, "tile_spec key").text;
          expect(TokKind::Equals, "'='");
          if (fk == "bm") ts.bm = parse_u32();
          else if (fk == "bn") ts.bn = parse_u32();
          else if (fk == "bk") ts.bk = parse_u32();
          else if (fk == "simds") ts.num_simdgroups = parse_u32();
          else if (fk == "dbl") ts.double_buffer = parse_bool();
          else if (fk == "split_k") ts.split_k = parse_u32();
          else throw std::runtime_error("Unknown tile_spec field: " + fk);
          if (check(TokKind::Comma)) consume();
        }
        expect(TokKind::RParen, "')'");
        hints.tile_spec = ts;
      } else if (k == "structure") {
        expect(TokKind::Equals, "'='");
        hints.structure = parse_structure_attr();
      } else if (k == "vec_width") {
        expect(TokKind::Equals, "'='");
        hints.vec_width = VecWidthAttr{parse_u32()};
      } else if (k == "layout_hint") {
        LayoutHintAttr lh;
        expect(TokKind::LParen, "'('");
        while (!check(TokKind::RParen)) {
          const std::string fk = expect(TokKind::Ident, "layout_hint key").text;
          expect(TokKind::Equals, "'='");
          if (fk == "layout") lh.layout = parse_tensor_layout_hint(expect(TokKind::Ident, "layout").text);
          else if (fk == "prefer_vec4") lh.prefer_vec4 = parse_bool();
          else throw std::runtime_error("Unknown layout_hint field: " + fk);
          if (check(TokKind::Comma)) consume();
        }
        expect(TokKind::RParen, "')'");
        hints.layout_hint = lh;
      } else if (k == "persistent_grid") {
        PersistentGridAttr pg;
        expect(TokKind::LParen, "'('");
        while (!check(TokKind::RParen)) {
          const std::string fk = expect(TokKind::Ident, "persistent_grid key").text;
          expect(TokKind::Equals, "'='");
          if (fk == "enabled") pg.enabled = parse_bool();
          else if (fk == "max_tgs") pg.max_tgs_per_core = parse_u32();
          else if (fk == "swizzle_w") pg.swizzle_w = parse_u32();
          else throw std::runtime_error("Unknown persistent_grid field: " + fk);
          if (check(TokKind::Comma)) consume();
        }
        expect(TokKind::RParen, "')'");
        hints.persistent_grid = pg;
      } else {
        throw std::runtime_error("Unknown hint key: " + k);
      }
      if (check(TokKind::Comma)) consume();
    }
    expect(TokKind::RBrace, "'}'");
    return hints;
  }

  std::vector<int64_t> parse_i64_list() {
    expect(TokKind::LBracket, "'['");
    std::vector<int64_t> out;
    while (!check(TokKind::RBracket)) {
      out.push_back(parse_i64());
      if (check(TokKind::Comma)) consume();
    }
    expect(TokKind::RBracket, "']'");
    return out;
  }

  std::vector<int32_t> parse_i32_list() {
    auto v = parse_i64_list();
    std::vector<int32_t> out;
    out.reserve(v.size());
    for (int64_t x : v) out.push_back(static_cast<int32_t>(x));
    return out;
  }

  std::vector<uint32_t> parse_u32_list() {
    expect(TokKind::LBracket, "'['");
    std::vector<uint32_t> out;
    while (!check(TokKind::RBracket)) {
      out.push_back(parse_u32());
      if (check(TokKind::Comma)) consume();
    }
    expect(TokKind::RBracket, "']'");
    return out;
  }

  ValueId lookup_value(const std::unordered_map<std::string, ValueId>& scope, const Token& tok) const {
    auto it = scope.find(tok.text);
    if (it == scope.end()) {
      throw std::runtime_error("Undefined value '%" + tok.text + "' at line " + std::to_string(tok.line));
    }
    return it->second;
  }

  void parse_kernel(Module& mod) {
    expect_ident("kernel");
    const std::string kname = expect(TokKind::SymbolRef, "@kernel_name").text;
    Kernel& kernel = mod.add_kernel(kname);

    std::unordered_map<std::string, ValueId> scope;
    expect(TokKind::LParen, "'('");
    while (!check(TokKind::RParen)) {
      const std::string pname = expect(TokKind::ValueRef, "%param").text;
      expect(TokKind::Colon, "':'");
      const Type pty = parse_type();
      KernelParam param;
      param.name = pname;
      param.type = pty;
      if (check(TokKind::LBrace)) {
        consume();
        while (!check(TokKind::RBrace)) {
          const std::string k = expect(TokKind::Ident, "param attr").text;
          expect(TokKind::Equals, "'='");
          if (k == "constexpr") {
            param.is_constexpr = true;
            param.constexpr_val = parse_i64();
          } else if (k == "divisibility") {
            param.divisibility = parse_u32();
          } else if (k == "writable") {
            param.writable = parse_bool();
          } else {
            throw std::runtime_error("Unknown param attribute: " + k);
          }
          if (check(TokKind::Comma)) consume();
        }
        expect(TokKind::RBrace, "'}'");
      }
      const uint32_t arg_idx = static_cast<uint32_t>(kernel.block(kernel.entry_block).args.size());
      kernel.params.push_back(param);
      const ValueId vid = kernel.create_value(pty, kInvalidId, 0, kernel.entry_block, arg_idx, pname);
      kernel.block(kernel.entry_block).args.push_back(vid);
      scope[pname] = vid;
      if (check(TokKind::Comma)) consume();
    }
    expect(TokKind::RParen, "')'");

    kernel.hints = parse_optional_hints();
    expect(TokKind::LBrace, "'{'");
    parse_block_body(kernel, kernel.entry_block, scope);
    expect(TokKind::RBrace, "'}'");
  }

  void parse_block_body(Kernel& kernel, BlockId block_id, std::unordered_map<std::string, ValueId> scope) {
    while (!check(TokKind::RBrace) && !check(TokKind::Eof)) {
      std::vector<std::string> res_names;
      if (check(TokKind::ValueRef)) {
        res_names.push_back(consume().text);
        while (check(TokKind::Comma)) {
          consume();
          res_names.push_back(expect(TokKind::ValueRef, "%result").text);
        }
        expect(TokKind::Equals, "'='");
      }

      const Token& op_tok = expect(TokKind::Ident, "op name");
      auto kind_opt = parse_op_kind(op_tok.text);
      if (!kind_opt) {
        throw std::runtime_error("Unknown op '" + op_tok.text + "' at line " + std::to_string(op_tok.line));
      }
      const OpKind kind = *kind_opt;

      if (kind == OpKind::ForLoop) {
        parse_for_loop(kernel, block_id, res_names, scope);
        continue;
      }
      if (kind == OpKind::If) {
        parse_if(kernel, block_id, res_names, scope);
        continue;
      }

      OpPayload attr = EmptyAttr{};
      if (kind == OpKind::Unary) {
        expect(TokKind::Dot, "'.'");
        const std::string sub = expect(TokKind::Ident, "unary op").text;
        auto uop = parse_unary_op(sub);
        if (!uop) throw std::runtime_error("Unknown unary op: " + sub);
        attr = UnaryAttr{*uop};
      } else if (kind == OpKind::Binary) {
        expect(TokKind::Dot, "'.'");
        const std::string sub = expect(TokKind::Ident, "binary op").text;
        auto bop = parse_binary_op(sub);
        if (!bop) throw std::runtime_error("Unknown binary op: " + sub);
        attr = BinaryAttr{*bop};
      }

      std::vector<ValueId> operands;
      if (check(TokKind::ValueRef) && peek().line == op_tok.line) {
        operands.push_back(lookup_value(scope, consume()));
        while (check(TokKind::Comma) && peek().line == op_tok.line) {
          consume();
          operands.push_back(lookup_value(scope, expect(TokKind::ValueRef, "%operand")));
        }
      }

      // Raw const token string is parsed after result type so variant type is exact.
      std::optional<std::string> raw_const_val;
      if (check(TokKind::LBrace)) {
        attr = parse_op_attr(kind, raw_const_val);
      } else if (kind == OpKind::Load || kind == OpKind::Store || kind == OpKind::AtomicAdd) {
        attr = LoadStoreAttr{};
      } else if (kind == OpKind::Dot) {
        attr = DotAttr{};
      } else if (kind == OpKind::Barrier) {
        attr = BarrierAttr{};
      }

      OpHints hints = parse_optional_hints();

      std::vector<Type> result_types;
      if (check(TokKind::Colon)) {
        consume();
        result_types = parse_type_or_tuple();
      }

      if (res_names.size() != result_types.size()) {
        throw std::runtime_error("Result name count (" + std::to_string(res_names.size()) +
                                 ") does not match result type count (" + std::to_string(result_types.size()) +
                                 ") at line " + std::to_string(op_tok.line));
      }

      if (kind == OpKind::Constant && raw_const_val.has_value() && result_types.size() == 1 &&
          result_types[0].is_scalar()) {
        const ScalarDType dt = result_types[0].as_scalar();
        const std::string& raw = *raw_const_val;
        ConstAttr ca;
        if (dt == ScalarDType::Bool) {
          ca.val = (raw == "true");
        } else if (dt == ScalarDType::U32) {
          std::string stripped = (!raw.empty() && raw.back() == 'u') ? raw.substr(0, raw.size() - 1) : raw;
          ca.val = static_cast<uint64_t>(std::stoull(stripped));
        } else if (dt == ScalarDType::I32 || dt == ScalarDType::I64) {
          ca.val = static_cast<int64_t>(std::stoll(raw));
        } else {
          if (raw == "nan") ca.val = std::numeric_limits<double>::quiet_NaN();
          else if (raw == "inf") ca.val = std::numeric_limits<double>::infinity();
          else if (raw == "-inf") ca.val = -std::numeric_limits<double>::infinity();
          else ca.val = std::stod(raw);
        }
        attr = ca;
      }

      const OpId op_id = kernel.create_op(kind, block_id, std::move(operands), result_types, std::move(attr),
                                          std::move(hints));
      for (size_t i = 0; i < res_names.size(); ++i) {
        scope[res_names[i]] = kernel.op(op_id).results[i];
      }
    }
  }

  OpPayload parse_op_attr(OpKind kind, std::optional<std::string>& raw_const_val) {
    expect(TokKind::LBrace, "'{'");
    OpPayload result = EmptyAttr{};

    if (kind == OpKind::Constant) {
      expect_ident("val");
      expect(TokKind::Equals, "'='");
      const Token& t = consume();
      raw_const_val = t.text;
      expect(TokKind::RBrace, "'}'");
      return ConstAttr{};
    }

    if (kind == OpKind::ProgramId || kind == OpKind::NumPrograms || kind == OpKind::ExpandDims) {
      AxisAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "axis key").text;
        expect(TokKind::Equals, "'='");
        if (k == "axis") a.axis = parse_u32();
        else throw std::runtime_error("Unknown AxisAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::Arange) {
      ArangeAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "arange key").text;
        expect(TokKind::Equals, "'='");
        if (k == "start") a.start = parse_i64();
        else if (k == "end") a.end = parse_i64();
        else throw std::runtime_error("Unknown ArangeAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::Permute) {
      PermuteAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "permute key").text;
        expect(TokKind::Equals, "'='");
        if (k == "perm") a.perm = parse_u32_list();
        else throw std::runtime_error("Unknown PermuteAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::Load || kind == OpKind::Store || kind == OpKind::AtomicAdd) {
      LoadStoreAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "load/store key").text;
        expect(TokKind::Equals, "'='");
        if (k == "align") a.alignment_bytes = parse_u32();
        else if (k == "vec") a.vec_width = parse_u32();
        else if (k == "has_mask") a.has_mask = parse_bool();
        else if (k == "has_other") a.has_other = parse_bool();
        else if (k == "volatile") a.is_volatile = parse_bool();
        else if (k == "nt") a.non_temporal = parse_bool();
        else throw std::runtime_error("Unknown LoadStoreAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::Dot) {
      DotAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "dot key").text;
        expect(TokKind::Equals, "'='");
        if (k == "acc") {
          const std::string dt_str = expect(TokKind::Ident, "dtype").text;
          auto dt = parse_dtype(dt_str);
          if (!dt) throw std::runtime_error("Unknown acc dtype: " + dt_str);
          a.acc_dtype = *dt;
        } else if (k == "tf32") a.allow_tf32 = parse_bool();
        else if (k == "ta") a.transpose_a = parse_bool();
        else if (k == "tb") a.transpose_b = parse_bool();
        else if (k == "lhs_struct") a.lhs_struct = parse_structure_attr();
        else if (k == "rhs_struct") a.rhs_struct = parse_structure_attr();
        else if (k == "out_struct") a.out_struct = parse_structure_attr();
        else throw std::runtime_error("Unknown DotAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::Reduce) {
      ReduceAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "reduce key").text;
        expect(TokKind::Equals, "'='");
        if (k == "kind") a.kind = parse_reduce_kind(expect(TokKind::Ident, "reduce kind").text);
        else if (k == "mode") a.result_mode = parse_reduce_mode(expect(TokKind::Ident, "reduce mode").text);
        else if (k == "axis") a.axis = static_cast<int32_t>(parse_i64());
        else if (k == "keep_dims") a.keep_dims = parse_bool();
        else throw std::runtime_error("Unknown ReduceAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::Scan) {
      ScanAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "scan key").text;
        expect(TokKind::Equals, "'='");
        if (k == "kind") a.kind = parse_scan_kind(expect(TokKind::Ident, "scan kind").text);
        else if (k == "axis") a.axis = static_cast<int32_t>(parse_i64());
        else if (k == "exclusive") a.exclusive = parse_bool();
        else if (k == "reverse") a.reverse = parse_bool();
        else if (k == "has_mask") a.has_mask = parse_bool();
        else throw std::runtime_error("Unknown ScanAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::ImplicitGemmConv) {
      ConvAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "conv key").text;
        expect(TokKind::Equals, "'='");
        if (k == "ndim") a.ndim = parse_u32();
        else if (k == "stride") a.stride = parse_i32_list();
        else if (k == "padding") a.padding = parse_i32_list();
        else if (k == "dilation") a.dilation = parse_i32_list();
        else if (k == "groups") a.groups = static_cast<int32_t>(parse_i64());
        else if (k == "transposed") a.transposed = parse_bool();
        else if (k == "layout") a.layout = parse_tensor_layout_hint(expect(TokKind::Ident, "layout").text);
        else throw std::runtime_error("Unknown ConvAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::FlashAttentionTile) {
      FlashAttnAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "flash_attn key").text;
        expect(TokKind::Equals, "'='");
        if (k == "scale") a.scale = static_cast<float>(parse_double());
        else if (k == "causal") a.causal = parse_bool();
        else if (k == "causal_offset") a.causal_offset = static_cast<int32_t>(parse_i64());
        else throw std::runtime_error("Unknown FlashAttnAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::ThreadgroupAlloc) {
      ThreadgroupAllocAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "tg_alloc key").text;
        expect(TokKind::Equals, "'='");
        if (k == "elem") {
          const std::string dt_str = expect(TokKind::Ident, "dtype").text;
          auto dt = parse_dtype(dt_str);
          if (!dt) throw std::runtime_error("Unknown elem dtype: " + dt_str);
          a.elem = *dt;
        } else if (k == "shape") a.shape = parse_i64_list();
        else if (k == "swizzle") a.swizzle = parse_swizzle_mode(expect(TokKind::Ident, "swizzle").text);
        else if (k == "pad") a.pad_elems = parse_u32();
        else if (k == "stages") a.stages = parse_u32();
        else throw std::runtime_error("Unknown ThreadgroupAllocAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::ThreadgroupStage) {
      ThreadgroupStageAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "tg_stage key").text;
        expect(TokKind::Equals, "'='");
        if (k == "stage") a.stage_idx = parse_u32();
        else if (k == "zero_pad") a.zero_pad_oob = parse_bool();
        else throw std::runtime_error("Unknown ThreadgroupStageAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::Barrier) {
      BarrierAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "barrier key").text;
        expect(TokKind::Equals, "'='");
        if (k == "scope") a.scope = parse_barrier_scope(expect(TokKind::Ident, "scope").text);
        else throw std::runtime_error("Unknown BarrierAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    if (kind == OpKind::SimdgroupMatmul) {
      SimdgroupMatmulAttr a;
      while (!check(TokKind::RBrace)) {
        const std::string k = expect(TokKind::Ident, "sgmm key").text;
        expect(TokKind::Equals, "'='");
        if (k == "m_tiles") a.m_tiles = parse_u32();
        else if (k == "n_tiles") a.n_tiles = parse_u32();
        else if (k == "k_step") a.k_step = parse_u32();
        else if (k == "ldm_a") a.ldm_a = parse_u32();
        else if (k == "ldm_b") a.ldm_b = parse_u32();
        else if (k == "acc") {
          const std::string dt_str = expect(TokKind::Ident, "dtype").text;
          auto dt = parse_dtype(dt_str);
          if (!dt) throw std::runtime_error("Unknown acc dtype: " + dt_str);
          a.acc_dtype = *dt;
        } else throw std::runtime_error("Unknown SimdgroupMatmulAttr key: " + k);
        if (check(TokKind::Comma)) consume();
      }
      expect(TokKind::RBrace, "'}'");
      return a;
    }

    throw std::runtime_error("Unexpected attribute block for op " + std::string(op_kind_name(kind)));
  }

  void parse_for_loop(Kernel& kernel, BlockId parent_block, const std::vector<std::string>& res_names,
                      std::unordered_map<std::string, ValueId>& scope) {
    expect(TokKind::LParen, "'('");
    const std::string iv_name = expect(TokKind::ValueRef, "%iv").text;
    expect(TokKind::Colon, "':'");
    const Type iv_ty = parse_type();

    struct IterArgSpec {
      std::string name;
      Type type;
      ValueId init_val;
    };
    std::vector<IterArgSpec> iter_specs;
    while (check(TokKind::Comma)) {
      consume();
      const std::string aname = expect(TokKind::ValueRef, "%iter_arg").text;
      expect(TokKind::Colon, "':'");
      const Type aty = parse_type();
      expect(TokKind::Equals, "'='");
      const ValueId init_val = lookup_value(scope, expect(TokKind::ValueRef, "%init"));
      iter_specs.push_back(IterArgSpec{aname, aty, init_val});
    }
    expect(TokKind::RParen, "')'");

    expect_ident("in");
    expect(TokKind::LParen, "'('");
    const ValueId lo = lookup_value(scope, expect(TokKind::ValueRef, "%lo"));
    expect(TokKind::Comma, "','");
    const ValueId hi = lookup_value(scope, expect(TokKind::ValueRef, "%hi"));
    expect(TokKind::Comma, "','");
    const ValueId step = lookup_value(scope, expect(TokKind::ValueRef, "%step"));
    expect(TokKind::RParen, "')'");

    OpHints hints = parse_optional_hints();

    std::vector<Type> result_types;
    if (check(TokKind::Colon)) {
      consume();
      result_types = parse_type_or_tuple();
    }
    if (res_names.size() != result_types.size()) {
      throw std::runtime_error("ForLoop result name count does not match result type count");
    }

    std::vector<ValueId> operands{lo, hi, step};
    for (const auto& s : iter_specs) {
      operands.push_back(s.init_val);
    }

    const OpId op_id =
        kernel.create_op(OpKind::ForLoop, parent_block, std::move(operands), result_types, EmptyAttr{}, std::move(hints));
    const BlockId body_id = kernel.create_block(op_id);
    kernel.op(op_id).regions.push_back(body_id);

    auto body_scope = scope;
    const ValueId iv_val = kernel.create_value(iv_ty, kInvalidId, 0, body_id, 0, iv_name);
    kernel.block(body_id).args.push_back(iv_val);
    body_scope[iv_name] = iv_val;

    for (size_t i = 0; i < iter_specs.size(); ++i) {
      const ValueId arg_val =
          kernel.create_value(iter_specs[i].type, kInvalidId, 0, body_id, static_cast<uint32_t>(i + 1), iter_specs[i].name);
      kernel.block(body_id).args.push_back(arg_val);
      body_scope[iter_specs[i].name] = arg_val;
    }

    expect(TokKind::LBrace, "'{'");
    parse_block_body(kernel, body_id, body_scope);
    expect(TokKind::RBrace, "'}'");

    for (size_t i = 0; i < res_names.size(); ++i) {
      scope[res_names[i]] = kernel.op(op_id).results[i];
    }
  }

  void parse_if(Kernel& kernel, BlockId parent_block, const std::vector<std::string>& res_names,
                std::unordered_map<std::string, ValueId>& scope) {
    const ValueId cond = lookup_value(scope, expect(TokKind::ValueRef, "%cond"));
    OpHints hints = parse_optional_hints();

    std::vector<Type> result_types;
    if (check(TokKind::Colon)) {
      consume();
      result_types = parse_type_or_tuple();
    }
    if (res_names.size() != result_types.size()) {
      throw std::runtime_error("If result name count does not match result type count");
    }

    const OpId op_id =
        kernel.create_op(OpKind::If, parent_block, {cond}, result_types, EmptyAttr{}, std::move(hints));
    const BlockId then_id = kernel.create_block(op_id);
    kernel.op(op_id).regions.push_back(then_id);

    expect(TokKind::LBrace, "'{'");
    parse_block_body(kernel, then_id, scope);
    expect(TokKind::RBrace, "'}'");

    if (check_ident("else")) {
      consume();
      const BlockId else_id = kernel.create_block(op_id);
      kernel.op(op_id).regions.push_back(else_id);
      expect(TokKind::LBrace, "'{'");
      parse_block_body(kernel, else_id, scope);
      expect(TokKind::RBrace, "'}'");
    }

    for (size_t i = 0; i < res_names.size(); ++i) {
      scope[res_names[i]] = kernel.op(op_id).results[i];
    }
  }

  std::vector<Token> tokens_;
  size_t pos_ = 0;
};

}  // namespace

ParseResult IRParser::parse(std::string_view text) {
  ParseResult res;
  try {
    Lexer lexer(text);
    auto tokens = lexer.tokenize();
    Parser parser(std::move(tokens));
    res.module = parser.parse_module();
    res.ok = true;
  } catch (const std::exception& e) {
    res.ok = false;
    res.error = e.what();
    res.module.reset();
  }
  return res;
}

}  // namespace m1triton::ir
