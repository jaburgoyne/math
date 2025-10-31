#pragma once
#include <cstdio>
#include <mutex>
#include <string>
#include <string_view>
#include <chrono>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <vector>

class JsonLogger {
public:
    enum class Level { Trace, Debug, Info, Warn, Error, Critical };

    JsonLogger() : out_(stderr), min_(Level::Info), auto_flush_(true) {
      this->set_file("../laplace_logger.jsonl", false);
    }
    ~JsonLogger() {
        if (out_ && out_ != stderr) std::fclose(out_);
    }
    void set_min_level(Level l) { min_ = l; }
    void set_auto_flush(bool b) { auto_flush_ = b; }
    bool set_file(const char* path, bool append = true) {
        std::lock_guard<std::mutex> lk(m_);
        FILE* f = std::fopen(path, append ? "ab" : "wb");
        if (!f) return false;
        if (out_ && out_ != stderr) std::fclose(out_);
        out_ = f;
        return true;
    }
    void set_stream(FILE* f) {
        std::lock_guard<std::mutex> lk(m_);
        if (out_ && out_ != stderr) std::fclose(out_);
        out_ = f ? f : stderr;
    }

    template <typename... Args>
    void log(Level lvl, std::string_view msg, Args&&... args) {
        static_assert(sizeof...(Args) % 2 == 0, "Key/value args must be in pairs.");
        if (lvl < min_) return;
        std::string line;
        line.reserve(256);
        line.push_back('{');
        bool first = true;
        append_field(line, first, "ts", current_ts_iso());
        append_field(line, first, "level", level_name(lvl));
        append_field(line, first, "msg", msg);
        write_pairs(line, first, std::forward<Args>(args)...);
        line.push_back('}'); line.push_back('\n');
        write_line(line);
    }

    template <typename... Args>
    void log_now(Level lvl, std::string_view msg, Args&&... args) {
        static_assert(sizeof...(Args) % 2 == 0, "Key/value args must be in pairs.");
        if (lvl < min_) return;
        std::string line;
        line.reserve(256);
        line.push_back('{');
        bool first = true;
        append_field(line, first, "ts", current_ts_iso());
        append_field(line, first, "hrtime_ns", now_ns());
        append_field(line, first, "level", level_name(lvl));
        append_field(line, first, "msg", msg);
        write_pairs(line, first, std::forward<Args>(args)...);
        line.push_back('}'); line.push_back('\n');
        write_line(line);
    }

    template <typename T, typename... Args>
    void log_vec(Level lvl, std::string_view msg, std::string_view key, const std::vector<T>& v, Args&&... rest) {
        static_assert(sizeof...(Args) % 2 == 0, "Key/value args must be in pairs.");
        if (lvl < min_) return;
        std::string line;
        line.reserve(256);
        line.push_back('{');
        bool first = true;
        append_field(line, first, "ts", current_ts_iso());
        append_field(line, first, "level", level_name(lvl));
        append_field(line, first, "msg", msg);
        append_field(line, first, key, v);
        write_pairs(line, first, std::forward<Args>(rest)...);
        line.push_back('}'); line.push_back('\n');
        write_line(line);
    }

    template<typename... A> void trace(std::string_view m, A&&... a){ log(Level::Trace,    m, std::forward<A>(a)...); }
    template<typename... A> void debug(std::string_view m, A&&... a){ log(Level::Debug,    m, std::forward<A>(a)...); }
    template<typename... A> void info (std::string_view m, A&&... a){ log(Level::Info,     m, std::forward<A>(a)...); }
    template<typename... A> void warn (std::string_view m, A&&... a){ log(Level::Warn,     m, std::forward<A>(a)...); }
    template<typename... A> void error(std::string_view m, A&&... a){ log(Level::Error,    m, std::forward<A>(a)...); }
    template<typename... A> void critical(std::string_view m, A&&... a){ log(Level::Critical, m, std::forward<A>(a)...); }

    template<typename... A> void trace_now(std::string_view m, A&&... a){ log_now(Level::Trace,    m, std::forward<A>(a)...); }
    template<typename... A> void debug_now(std::string_view m, A&&... a){ log_now(Level::Debug,    m, std::forward<A>(a)...); }
    template<typename... A> void info_now (std::string_view m, A&&... a){ log_now(Level::Info,     m, std::forward<A>(a)...); }
    template<typename... A> void warn_now (std::string_view m, A&&... a){ log_now(Level::Warn,     m, std::forward<A>(a)...); }
    template<typename... A> void error_now(std::string_view m, A&&... a){ log_now(Level::Error,    m, std::forward<A>(a)...); }
    template<typename... A> void critical_now(std::string_view m, A&&... a){ log_now(Level::Critical, m, std::forward<A>(a)...); }

    template<typename T, typename... A> void trace_vec(std::string_view m, std::string_view k, const std::vector<T>& v, A&&... a){ log_vec(Level::Trace,    m, k, v, std::forward<A>(a)...); }
    template<typename T, typename... A> void debug_vec(std::string_view m, std::string_view k, const std::vector<T>& v, A&&... a){ log_vec(Level::Debug,    m, k, v, std::forward<A>(a)...); }
    template<typename T, typename... A> void info_vec (std::string_view m, std::string_view k, const std::vector<T>& v, A&&... a){ log_vec(Level::Info,     m, k, v, std::forward<A>(a)...); }
    template<typename T, typename... A> void warn_vec (std::string_view m, std::string_view k, const std::vector<T>& v, A&&... a){ log_vec(Level::Warn,     m, k, v, std::forward<A>(a)...); }
    template<typename T, typename... A> void error_vec(std::string_view m, std::string_view k, const std::vector<T>& v, A&&... a){ log_vec(Level::Error,    m, k, v, std::forward<A>(a)...); }
    template<typename T, typename... A> void critical_vec(std::string_view m, std::string_view k, const std::vector<T>& v, A&&... a){ log_vec(Level::Critical, m, k, v, std::forward<A>(a)...); }

    class Builder {
    public:
        Builder() { first_stack_.push_back(true); }
        template<typename T>
        Builder& field(std::string_view key, T&& value) {
            append_comma();
            escape_json(key, out_);
            out_.push_back(':');
            write_value(out_, std::forward<T>(value));
            return *this;
        }
        Builder& begin_object(std::string_view key) {
            append_comma();
            escape_json(key, out_);
            out_ += ":{";
            first_stack_.push_back(true);
            close_stack_.push_back('}');
            return *this;
        }
        Builder& begin_object() {
            append_comma();
            out_.push_back('{');
            first_stack_.push_back(true);
            close_stack_.push_back('}');
            return *this;
        }
        Builder& begin_array(std::string_view key) {
            append_comma();
            escape_json(key, out_);
            out_ += ":[";
            first_stack_.push_back(true);
            close_stack_.push_back(']');
            return *this;
        }
        Builder& begin_array() {
            append_comma();
            out_.push_back('[');
            first_stack_.push_back(true);
            close_stack_.push_back(']');
            return *this;
        }
        template<typename T>
        Builder& elem(T&& value) {
            append_comma();
            write_value(out_, std::forward<T>(value));
            return *this;
        }
        Builder& end() {
            if (close_stack_.empty()) return *this;
            out_.push_back(close_stack_.back());
            close_stack_.pop_back();
            first_stack_.pop_back();
            return *this;
        }
        bool balanced() const { return close_stack_.empty(); }
        const std::string& top_members() const { return out_; }
    private:
        std::string out_;
        std::vector<bool> first_stack_;
        std::vector<char> close_stack_;
        void append_comma() {
            auto&& first = first_stack_.back();
            if (!first) out_.push_back(',');
            first = false;
        }
        friend class JsonLogger;
    };
    std::pair<std::string, std::string> builder_init_vals{};
    void init_builder(std::string_view key, std::string_view value) {
        builder_init_vals = {std::string(key), std::string(value)};
    }
    Builder builder() const {
      auto bb = Builder{};
      if (!builder_init_vals.first.empty()) {
        bb.field(builder_init_vals.first, builder_init_vals.second);
      }
      return bb;
    }

    void commit(Level lvl, std::string_view msg, const Builder& b) {
        if (lvl < min_) return;
        std::string line;
        line.reserve(256 + b.top_members().size());
        line.push_back('{');
        bool first = true;
        append_field(line, first, "ts", current_ts_iso());
        append_field(line, first, "level", level_name(lvl));
        append_field(line, first, "msg", msg);
        if (!b.top_members().empty()) { line.push_back(','); line += b.top_members(); }
        line.push_back('}'); line.push_back('\n');
        write_line(line);
    }

    void commit_now(Level lvl, std::string_view msg, const Builder& b) {
        std::string line;
        line.reserve(256 + b.top_members().size());
        line.push_back('{');
        bool first = true;
        append_field(line, first, "ts", current_ts_iso());
        append_field(line, first, "hrtime_ns", now_ns());
        append_field(line, first, "level", level_name(lvl));
        append_field(line, first, "msg", msg);
        if (!b.top_members().empty()) { line.push_back(','); line += b.top_members(); }
        line.push_back('}'); line.push_back('\n');
        write_line(line);
    }

    static JsonLogger& instance() { static JsonLogger inst; return inst; }

private:
    FILE* out_;
    std::mutex m_;
    Level min_;
    bool auto_flush_;

    void write_line(const std::string& s) {
        std::fwrite(s.data(), 1, s.size(), out_);
        if (auto_flush_) std::fflush(out_);
    }

    static const char* level_name(Level l) {
        switch (l) {
            case Level::Trace:    return "trace";
            case Level::Debug:    return "debug";
            case Level::Info:     return "info";
            case Level::Warn:     return "warn";
            case Level::Error:    return "error";
            case Level::Critical: return "critical";
        }
        return "info";
    }

    static void escape_json(std::string_view s, std::string& out) {
        out.push_back('"');
        for (char ch : s) {
            unsigned char c = static_cast<unsigned char>(ch);
            switch (c) {
                case '\"': out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (c < 0x20) {
                        char buf[7];
                        std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                        out += buf;
                    } else {
                        out.push_back(static_cast<char>(c));
                    }
            }
        }
        out.push_back('"');
    }

    static std::string current_ts_iso() {
        using namespace std::chrono;
        const auto now  = system_clock::now();
        const auto secs = time_point_cast<seconds>(now);
        const auto ms   = duration_cast<milliseconds>(now - secs).count();
        std::time_t tt = system_clock::to_time_t(secs);
        std::tm tm;
    #if defined(_WIN32)
        gmtime_s(&tm, &tt);
    #else
        gmtime_r(&tt, &tm);
    #endif
        char buf[64];
        std::snprintf(buf, sizeof(buf),
                      "%04d-%02d-%02dT%02d:%02d:%02d.%03lldZ",
                      tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                      tm.tm_hour, tm.tm_min, tm.tm_sec,
                      static_cast<long long>(ms));
        return std::string(buf);
    }
    static long long now_ns() {
        using namespace std::chrono;
        return duration_cast<nanoseconds>(
            high_resolution_clock::now().time_since_epoch()).count();
    }

    static void append_comma(std::string& out, bool& first) {
        if (!first) out.push_back(',');
        else first = false;
    }
    template<typename K, typename V>
    static void append_field(std::string& out, bool& first, K&& key, V&& value) {
        append_comma(out, first);
        if constexpr (std::is_convertible_v<K, std::string_view>) {
            escape_json(std::string_view(std::forward<K>(key)), out);
        } else {
            static_assert(std::is_convertible_v<K, std::string_view>, "Key must be convertible to std::string_view");
        }
        out.push_back(':');
        write_value(out, std::forward<V>(value));
    }

    static void write_value(std::string& out, std::nullptr_t) { out += "null"; }
    static void write_value(std::string& out, bool v)         { out += v ? "true" : "false"; }
    static void write_value(std::string& out, const char* s)  { escape_json(std::string_view(s ? s : ""), out); }
    static void write_value(std::string& out, std::string_view s) { escape_json(s, out); }
    static void write_value(std::string& out, const std::string& s) { escape_json(std::string_view(s), out); }

    template<typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> && !std::is_same_v<T, char>>>
    static void write_value(std::string& out, T v) {
        std::ostringstream oss;
        oss.setf(std::ios::fmtflags(0), std::ios::floatfield);
        oss << std::setprecision(15) << v;
        out += oss.str();
    }
    template<typename T>
    static void write_value(std::string& out, const std::vector<T>& vec) {
        out.push_back('[');
        bool first = true;
        for (const auto& e : vec) {
            if (!first) out.push_back(',');
            first = false;
            write_value(out, e);
        }
        out.push_back(']');
    }
    static void write_value(std::string& out, const std::vector<bool>& vec) {
        out.push_back('[');
        bool first = true;
        for (bool b : vec) {
            if (!first) out.push_back(',');
            first = false;
            out += (b ? "true" : "false");
        }
        out.push_back(']');
    }
    template<class U> struct is_std_vector : std::false_type {};
    template<class U, class A> struct is_std_vector<std::vector<U,A>> : std::true_type {};
    template<class U> static constexpr bool is_std_vector_v = is_std_vector<std::decay_t<U>>::value;

    template<typename T>
    static std::enable_if_t<!std::is_arithmetic_v<T> &&
        !std::is_same_v<std::decay_t<T>, std::nullptr_t> &&
        !std::is_same_v<std::decay_t<T>, std::string> &&
        !std::is_same_v<std::decay_t<T>, std::string_view> &&
        !std::is_same_v<std::decay_t<T>, const char*> &&
        !std::is_same_v<std::decay_t<T>, bool> &&
        !std::is_same_v<std::decay_t<T>, std::vector<bool>> &&
        !is_std_vector_v<T>, void>
    write_value(std::string&, const T&) {
        static_assert(sizeof(T) == 0, "Unsupported value type for JSON logger.");
    }

    static void write_pairs(std::string&, bool&) {}
    template<typename K, typename V, typename... Rest>
    static void write_pairs(std::string& out, bool& first, K&& k, V&& v, Rest&&... rest) {
        append_field(out, first, std::forward<K>(k), std::forward<V>(v));
        if constexpr (sizeof...(rest) > 0) write_pairs(out, first, std::forward<Rest>(rest)...);
    }
};

inline JsonLogger& JLOG() { return JsonLogger::instance(); }
