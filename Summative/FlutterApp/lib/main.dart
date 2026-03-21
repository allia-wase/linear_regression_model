import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

/// API base URL. Replace with your Render deployment URL, e.g.:
/// https://your-api-name.onrender.com
const String apiBaseUrl = 'https://your-api-name.onrender.com';

void main() {
  runApp(const DepressionPredictorApp());
}

class DepressionPredictorApp extends StatelessWidget {
  const DepressionPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Depression Risk Predictor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false;
  String? _resultMessage;
  bool _isError = false;

  final _controllers = <String, TextEditingController>{};
  static const _fields = [
    ('age', 'Age (18-59)', '29', TextInputType.number),
    ('gender', 'Gender (Male/Female)', 'Male', TextInputType.text),
    ('academic_pressure', 'Academic Pressure (0-5)', '2', TextInputType.number),
    ('work_pressure', 'Work Pressure (0-5)', '0', TextInputType.number),
    ('cgpa', 'CGPA (0-10)', '7.5', TextInputType.number),
    ('study_satisfaction', 'Study Satisfaction (0-5)', '4', TextInputType.number),
    ('sleep_duration', 'Sleep Duration', '5-6 hours', TextInputType.text),
    ('dietary_habits', 'Dietary Habits', 'Healthy', TextInputType.text),
    ('degree', 'Degree', 'BSc', TextInputType.text),
    ('suicidal_thoughts', 'Suicidal Thoughts (Yes/No)', 'No', TextInputType.text),
    ('work_study_hours', 'Work/Study Hours (0-12)', '6', TextInputType.number),
    ('financial_stress', 'Financial Stress (1-5)', '2', TextInputType.number),
    ('family_history_mental_illness', 'Family History (Yes/No)', 'No', TextInputType.text),
  ];

  @override
  void initState() {
    super.initState();
    for (final f in _fields) {
      _controllers[f.$1] = TextEditingController(text: f.$3);
    }
  }

  @override
  void dispose() {
    for (final c in _controllers.values) {
      c.dispose();
    }
    super.dispose();
  }

  Future<void> _predict() async {
    if (_formKey.currentState?.validate() != true) return;

    setState(() {
      _isLoading = true;
      _resultMessage = null;
      _isError = false;
    });

    try {
      final body = <String, dynamic>{};
      for (final f in _fields) {
        final key = f.$1;
        final val = _controllers[key]!.text.trim();
        if (val.isEmpty) {
          setState(() {
            _isLoading = false;
            _resultMessage = 'Missing value: ${f.$2}';
            _isError = true;
          });
          return;
        }
        if (key == 'age' ||
            key == 'academic_pressure' ||
            key == 'work_pressure' ||
            key == 'cgpa' ||
            key == 'study_satisfaction' ||
            key == 'work_study_hours' ||
            key == 'financial_stress') {
          body[key] = double.tryParse(val) ?? val;
        } else {
          body[key] = val;
        }
      }

      final url = Uri.parse('$apiBaseUrl/predict');
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(body),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final pred = data['prediction'] as num;
        setState(() {
          _isLoading = false;
          _resultMessage = 'Predicted depression score: ${pred.toStringAsFixed(4)}\n'
              '(Higher = higher risk; 0-1 range)';
          _isError = false;
        });
      } else {
        final err = jsonDecode(response.body);
        final detail = err['detail'] ?? response.body;
        setState(() {
          _isLoading = false;
          _resultMessage = 'Error: $detail';
          _isError = true;
        });
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _resultMessage = 'Request failed: $e';
        _isError = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Depression Risk Predictor'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              ..._fields.map((f) => Padding(
                    padding: const EdgeInsets.only(bottom: 12),
                    child: TextFormField(
                      controller: _controllers[f.$1],
                      decoration: InputDecoration(
                        labelText: f.$2,
                        border: const OutlineInputBorder(),
                        filled: true,
                      ),
                      keyboardType: f.$4,
                      validator: (v) {
                        if (v == null || v.trim().isEmpty) {
                          return 'Required';
                        }
                        return null;
                      },
                    ),
                  )),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: _isLoading ? null : _predict,
                style: FilledButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                ),
                child: _isLoading
                    ? const SizedBox(
                        height: 24,
                        width: 24,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                    : const Text('Predict'),
              ),
              if (_resultMessage != null) ...[
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: _isError
                        ? Colors.red.shade50
                        : Colors.green.shade50,
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(
                      color: _isError ? Colors.red : Colors.green,
                      width: 1,
                    ),
                  ),
                  child: Text(
                    _resultMessage!,
                    style: TextStyle(
                      color: _isError ? Colors.red.shade900 : Colors.green.shade900,
                      fontSize: 16,
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
