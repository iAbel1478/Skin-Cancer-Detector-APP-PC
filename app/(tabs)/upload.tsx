import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, Alert, ScrollView, Platform, Dimensions } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Upload, Image as ImageIcon, FileText, CircleAlert as AlertCircle, Monitor, Smartphone } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import { useRouter } from 'expo-router';
import { useScans } from '@/hooks/useScans';

const { width } = Dimensions.get('window');
const isWeb = Platform.OS === 'web';
const isLargeScreen = width > 768;

export default function UploadScreen() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const router = useRouter();
  const { addScan } = useScans();

  const pickImage = async () => {
    const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (permissionResult.granted === false) {
      Alert.alert(
        'Permission Required', 
        `Please allow access to your ${isWeb ? 'files' : 'photo library'} to upload images.`
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  const analyzeImage = async () => {
    if (!selectedImage) return;
    
    setIsAnalyzing(true);
    
    try {
      // Add the uploaded image to scans
      await addScan({
        uri: selectedImage,
        result: 'analyzing',
        location: 'Uploaded image'
      });

      // Simulate analysis with longer time for desktop to show enhanced processing
      const analysisTime = isWeb ? 4000 : 3000;
      
      setTimeout(() => {
        setIsAnalyzing(false);
        Alert.alert(
          'Analysis Complete',
          `Image analyzed successfully! ${isWeb ? 'Enhanced desktop analysis' : 'Mobile analysis'} complete. Results have been saved to your scans.`,
          [
            {
              text: 'View Results',
              onPress: () => {
                setSelectedImage(null);
                router.push('/data');
              },
            },
            {
              text: 'Upload Another',
              onPress: () => setSelectedImage(null),
              style: 'cancel'
            }
          ]
        );
      }, analysisTime);
    } catch (error) {
      setIsAnalyzing(false);
      Alert.alert('Error', 'Failed to analyze image. Please try again.');
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView 
        showsVerticalScrollIndicator={false}
        contentContainerStyle={isLargeScreen ? styles.largeScreenContainer : undefined}
      >
        {/* Header */}
        <View style={[styles.header, isLargeScreen && styles.headerLarge]}>
          <View style={styles.headerContent}>
            <Text style={[styles.title, isLargeScreen && styles.titleLarge]}>
              Upload Image
            </Text>
            <Text style={[styles.subtitle, isLargeScreen && styles.subtitleLarge]}>
              Select a photo for AI analysis
              {isWeb ? ' - Drag & drop or browse files' : ''}
            </Text>
          </View>
          {isLargeScreen && (
            <View style={styles.platformIndicator}>
              {isWeb ? <Monitor size={24} color="#0066CC" /> : <Smartphone size={24} color="#0066CC" />}
            </View>
          )}
        </View>

        {/* Upload Area */}
        <View style={[styles.uploadSection, isLargeScreen && styles.uploadSectionLarge]}>
          {selectedImage ? (
            <View style={[styles.imageContainer, isLargeScreen && styles.imageContainerLarge]}>
              <Image 
                source={{ uri: selectedImage }} 
                style={[styles.selectedImage, isLargeScreen && styles.selectedImageLarge]} 
              />
              <TouchableOpacity 
                style={[styles.changeButton, isLargeScreen && styles.changeButtonLarge]} 
                onPress={pickImage}>
                <Text style={[styles.changeButtonText, isLargeScreen && styles.changeButtonTextLarge]}>
                  Change Image
                </Text>
              </TouchableOpacity>
            </View>
          ) : (
            <TouchableOpacity 
              style={[styles.uploadArea, isLargeScreen && styles.uploadAreaLarge]} 
              onPress={pickImage}>
              <Upload size={isLargeScreen ? 64 : 48} color="#0066CC" strokeWidth={2} />
              <Text style={[styles.uploadTitle, isLargeScreen && styles.uploadTitleLarge]}>
                {isWeb ? 'Select or Drop Image' : 'Select Image'}
              </Text>
              <Text style={[styles.uploadSubtitle, isLargeScreen && styles.uploadSubtitleLarge]}>
                Choose a clear photo of the skin lesion or mole
                {isWeb ? '\nSupports drag & drop for easy upload' : ''}
              </Text>
            </TouchableOpacity>
          )}
        </View>

        <View style={isLargeScreen ? styles.contentGrid : styles.contentStack}>
          {/* Guidelines */}
          <View style={[styles.guidelines, isLargeScreen && styles.guidelinesLarge]}>
            <Text style={[styles.guidelinesTitle, isLargeScreen && styles.guidelinesTitleLarge]}>
              Photo Guidelines
            </Text>
            
            <View style={[styles.guidelineItem, isLargeScreen && styles.guidelineItemLarge]}>
              <ImageIcon size={isLargeScreen ? 24 : 20} color="#10B981" strokeWidth={2} />
              <View style={styles.guidelineContent}>
                <Text style={[styles.guidelineItemTitle, isLargeScreen && styles.guidelineItemTitleLarge]}>
                  High Quality
                </Text>
                <Text style={[styles.guidelineItemText, isLargeScreen && styles.guidelineItemTextLarge]}>
                  Use clear, well-lit photos{isWeb ? ' with high resolution' : ''}
                </Text>
              </View>
            </View>

            <View style={[styles.guidelineItem, isLargeScreen && styles.guidelineItemLarge]}>
              <FileText size={isLargeScreen ? 24 : 20} color="#10B981" strokeWidth={2} />
              <View style={styles.guidelineContent}>
                <Text style={[styles.guidelineItemTitle, isLargeScreen && styles.guidelineItemTitleLarge]}>
                  Close-up Shot
                </Text>
                <Text style={[styles.guidelineItemText, isLargeScreen && styles.guidelineItemTextLarge]}>
                  Focus on the lesion or mole with clear detail
                </Text>
              </View>
            </View>

            <View style={[styles.guidelineItem, isLargeScreen && styles.guidelineItemLarge]}>
              <AlertCircle size={isLargeScreen ? 24 : 20} color="#F59E0B" strokeWidth={2} />
              <View style={styles.guidelineContent}>
                <Text style={[styles.guidelineItemTitle, isLargeScreen && styles.guidelineItemTitleLarge]}>
                  Medical Disclaimer
                </Text>
                <Text style={[styles.guidelineItemText, isLargeScreen && styles.guidelineItemTextLarge]}>
                  This tool is for screening purposes only. Always consult a dermatologist for proper diagnosis.
                </Text>
              </View>
            </View>
          </View>

          {/* Platform Features */}
          {isLargeScreen && (
            <View style={styles.platformFeatures}>
              <Text style={styles.platformFeaturesTitle}>
                {isWeb ? 'Desktop' : 'Mobile'} Features
              </Text>
              <View style={styles.featuresList}>
                {isWeb ? (
                  <>
                    <Text style={styles.featureItem}>• Drag & drop file upload</Text>
                    <Text style={styles.featureItem}>• Enhanced image processing</Text>
                    <Text style={styles.featureItem}>• Detailed analysis reports</Text>
                    <Text style={styles.featureItem}>• Export capabilities</Text>
                  </>
                ) : (
                  <>
                    <Text style={styles.featureItem}>• Quick photo selection</Text>
                    <Text style={styles.featureItem}>• Instant analysis</Text>
                    <Text style={styles.featureItem}>• Cloud sync</Text>
                    <Text style={styles.featureItem}>• Mobile notifications</Text>
                  </>
                )}
              </View>
            </View>
          )}
        </View>

        {/* Analyze Button */}
        {selectedImage && !isAnalyzing && (
          <View style={[styles.analyzeSection, isLargeScreen && styles.analyzeSectionLarge]}>
            <TouchableOpacity
              style={[styles.analyzeButton, isLargeScreen && styles.analyzeButtonLarge]}
              onPress={analyzeImage}>
              <Text style={[styles.analyzeButtonText, isLargeScreen && styles.analyzeButtonTextLarge]}>
                Analyze Image{isWeb ? ' (Enhanced)' : ''}
              </Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Analysis Progress */}
        {isAnalyzing && (
          <View style={[styles.analysisProgress, isLargeScreen && styles.analysisProgressLarge]}>
            <Text style={[styles.progressTitle, isLargeScreen && styles.progressTitleLarge]}>
              Analyzing Image
            </Text>
            <Text style={[styles.progressText, isLargeScreen && styles.progressTextLarge]}>
              Our AI is examining the image for potential signs of skin cancer. 
              {isWeb ? ' Enhanced desktop processing provides more detailed analysis.' : ' This may take a few moments.'}
            </Text>
            <View style={[styles.progressBar, isLargeScreen && styles.progressBarLarge]}>
              <View style={[styles.progressFill, isLargeScreen && styles.progressFillLarge]} />
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  largeScreenContainer: {
    maxWidth: 1200,
    alignSelf: 'center',
    width: '100%',
  },
  header: {
    padding: 24,
    paddingBottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLarge: {
    padding: 40,
    paddingBottom: 24,
  },
  headerContent: {
    flex: 1,
  },
  platformIndicator: {
    marginLeft: 16,
  },
  title: {
    fontSize: 28,
    fontFamily: 'Inter-Bold',
    color: '#111827',
    marginBottom: 4,
  },
  titleLarge: {
    fontSize: 36,
  },
  subtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
  },
  subtitleLarge: {
    fontSize: 18,
  },
  uploadSection: {
    padding: 24,
    paddingTop: 16,
  },
  uploadSectionLarge: {
    padding: 40,
    paddingTop: 24,
  },
  uploadArea: {
    backgroundColor: '#FFFFFF',
    borderWidth: 2,
    borderColor: '#E5E7EB',
    borderStyle: 'dashed',
    borderRadius: 16,
    padding: 48,
    alignItems: 'center',
    justifyContent: 'center',
  },
  uploadAreaLarge: {
    borderRadius: 24,
    padding: 80,
  },
  uploadTitle: {
    fontSize: 20,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginTop: 16,
    marginBottom: 8,
  },
  uploadTitleLarge: {
    fontSize: 28,
    marginTop: 24,
    marginBottom: 12,
  },
  uploadSubtitle: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    textAlign: 'center',
    lineHeight: 20,
  },
  uploadSubtitleLarge: {
    fontSize: 16,
    lineHeight: 24,
  },
  imageContainer: {
    alignItems: 'center',
  },
  imageContainerLarge: {
    padding: 20,
  },
  selectedImage: {
    width: 250,
    height: 250,
    borderRadius: 16,
    marginBottom: 16,
  },
  selectedImageLarge: {
    width: 400,
    height: 400,
    borderRadius: 24,
    marginBottom: 24,
  },
  changeButton: {
    backgroundColor: '#0066CC',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  changeButtonLarge: {
    paddingHorizontal: 32,
    paddingVertical: 16,
    borderRadius: 12,
  },
  changeButtonText: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#FFFFFF',
  },
  changeButtonTextLarge: {
    fontSize: 18,
  },
  contentGrid: {
    flexDirection: 'row',
    gap: 40,
    paddingHorizontal: 40,
  },
  contentStack: {
    flexDirection: 'column',
  },
  guidelines: {
    padding: 24,
    paddingTop: 16,
  },
  guidelinesLarge: {
    flex: 2,
    padding: 0,
  },
  guidelinesTitle: {
    fontSize: 20,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginBottom: 16,
  },
  guidelinesTitleLarge: {
    fontSize: 24,
    marginBottom: 24,
  },
  guidelineItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 16,
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  guidelineItemLarge: {
    padding: 24,
    borderRadius: 16,
    marginBottom: 20,
  },
  guidelineContent: {
    flex: 1,
    marginLeft: 12,
  },
  guidelineItemTitle: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginBottom: 4,
  },
  guidelineItemTitleLarge: {
    fontSize: 18,
    marginBottom: 8,
  },
  guidelineItemText: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    lineHeight: 20,
  },
  guidelineItemTextLarge: {
    fontSize: 16,
    lineHeight: 24,
  },
  platformFeatures: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    padding: 24,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
    height: 'fit-content',
  },
  platformFeaturesTitle: {
    fontSize: 20,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginBottom: 16,
  },
  featuresList: {
    gap: 8,
  },
  featureItem: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    lineHeight: 20,
  },
  analyzeSection: {
    padding: 24,
    paddingTop: 16,
  },
  analyzeSectionLarge: {
    padding: 40,
    paddingTop: 24,
  },
  analyzeButton: {
    backgroundColor: '#0066CC',
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  analyzeButtonLarge: {
    paddingVertical: 20,
    borderRadius: 16,
  },
  analyzeButtonText: {
    fontSize: 18,
    fontFamily: 'Inter-SemiBold',
    color: '#FFFFFF',
  },
  analyzeButtonTextLarge: {
    fontSize: 20,
  },
  analysisProgress: {
    margin: 24,
    padding: 24,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  analysisProgressLarge: {
    margin: 40,
    padding: 32,
    borderRadius: 20,
  },
  progressTitle: {
    fontSize: 18,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginBottom: 8,
    textAlign: 'center',
  },
  progressTitleLarge: {
    fontSize: 24,
    marginBottom: 12,
  },
  progressText: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    textAlign: 'center',
    lineHeight: 20,
    marginBottom: 16,
  },
  progressTextLarge: {
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 24,
  },
  progressBar: {
    height: 4,
    backgroundColor: '#E5E7EB',
    borderRadius: 2,
    overflow: 'hidden',
  },
  progressBarLarge: {
    height: 6,
    borderRadius: 3,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#0066CC',
    width: '60%',
  },
  progressFillLarge: {
    width: '75%',
  },
});