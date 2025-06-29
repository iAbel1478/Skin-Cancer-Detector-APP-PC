import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image, Alert, Modal, Dimensions, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Camera, Upload, Tag, Trash2, Eye, Plus, Monitor, Smartphone, CircleCheck as CheckCircle, TriangleAlert as AlertTriangle, X } from 'lucide-react-native';
import * as ImagePicker from 'expo-image-picker';
import { useScans, ScanResult } from '@/hooks/useScans';
import { useFocusEffect } from '@react-navigation/native';

const { width } = Dimensions.get('window');
const isWeb = Platform.OS === 'web';
const isLargeScreen = width > 768;

export default function DataScreen() {
  const { scans, addScan, deleteScan, isLoading, refreshScans } = useScans();
  const [selectedScan, setSelectedScan] = useState<ScanResult | null>(null);
  const [isViewingDetails, setIsViewingDetails] = useState(false);
  const [filter, setFilter] = useState<'all' | 'benign' | 'concerning'>('all');
  const [isDeleting, setIsDeleting] = useState<string | null>(null);

  // Refresh scans when component mounts or becomes focused
  useEffect(() => {
    console.log('Data screen mounted, refreshing scans');
    refreshScans();
  }, [refreshScans]);

  // Also refresh when the screen comes into focus (tab navigation)
  useFocusEffect(
    React.useCallback(() => {
      console.log('Data screen focused, refreshing scans');
      refreshScans();
    }, [refreshScans])
  );

  const addNewScan = async () => {
    try {
      const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();
      
      if (permissionResult.granted === false) {
        Alert.alert('Permission Required', 'Please allow access to your photo library.');
        return;
      }

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 0.8,
      });

      if (!result.canceled && result.assets[0]) {
        console.log('Adding new scan from gallery:', result.assets[0].uri);
        await addScan({
          uri: result.assets[0].uri,
          result: 'analyzing',
          location: 'Uploaded from gallery'
        });
        
        // Show immediate feedback
        Alert.alert(
          'Image Added!',
          'Your image has been added and is being analyzed. Results will appear shortly.',
          [{ text: 'OK' }]
        );
      }
    } catch (error) {
      console.error('Error adding scan:', error);
      Alert.alert('Error', 'Failed to add scan. Please try again.');
    }
  };

  const handleDeleteScan = async (scanId: string, event?: any) => {
    // Stop event propagation to prevent opening the modal
    if (event) {
      event.stopPropagation();
      event.preventDefault();
    }

    console.log('=== HANDLE DELETE SCAN START ===');
    console.log('Delete button pressed for scan:', scanId);
    
    // Prevent multiple delete operations on the same scan
    if (isDeleting === scanId) {
      console.log('Delete already in progress for scan:', scanId);
      return;
    }
    
    Alert.alert(
      'Delete Scan',
      'Are you sure you want to remove this scan from your records?',
      [
        { 
          text: 'Cancel', 
          style: 'cancel',
          onPress: () => {
            console.log('Delete cancelled by user');
            setIsDeleting(null);
          }
        },
        { 
          text: 'Delete', 
          style: 'destructive',
          onPress: async () => {
            try {
              console.log('User confirmed deletion of scan:', scanId);
              setIsDeleting(scanId);
              
              // Perform the deletion
              await deleteScan(scanId);
              console.log('Scan deleted successfully from storage');
              
              // Close modal if the deleted scan was being viewed
              if (selectedScan?.id === scanId) {
                setSelectedScan(null);
                setIsViewingDetails(false);
              }
              
              // Show success feedback
              Alert.alert(
                'Scan Deleted',
                'The scan has been removed from your records.',
                [{ 
                  text: 'OK',
                  onPress: () => {
                    console.log('Delete success dialog dismissed');
                  }
                }]
              );
              
            } catch (error) {
              console.error('Error deleting scan:', error);
              Alert.alert(
                'Delete Failed', 
                'Failed to delete scan. Please try again.',
                [{ text: 'OK' }]
              );
            } finally {
              setIsDeleting(null);
              console.log('=== HANDLE DELETE SCAN COMPLETE ===');
            }
          }
        }
      ]
    );
  };

  const handleScanPress = (scan: ScanResult) => {
    // Only open modal if not currently deleting this scan
    if (isDeleting !== scan.id) {
      setSelectedScan(scan);
      setIsViewingDetails(true);
    }
  };

  const filteredScans = scans.filter(scan => filter === 'all' || scan.result === filter);

  const stats = {
    total: scans.filter(s => s.result !== 'analyzing').length,
    benign: scans.filter(scan => scan.result === 'benign').length,
    concerning: scans.filter(scan => scan.result === 'concerning').length,
    analyzing: scans.filter(scan => scan.result === 'analyzing').length
  };

  const getResultColor = (result: string) => {
    switch (result) {
      case 'benign': return '#10B981';
      case 'concerning': return '#EF4444';
      case 'analyzing': return '#F59E0B';
      default: return '#6B7280';
    }
  };

  const getResultIcon = (result: string) => {
    switch (result) {
      case 'benign': return CheckCircle;
      case 'concerning': return AlertTriangle;
      case 'analyzing': return Eye;
      default: return CheckCircle;
    }
  };

  const getResultLabel = (result: string) => {
    switch (result) {
      case 'benign': return 'Looks Good';
      case 'concerning': return 'See Doctor';
      case 'analyzing': return 'Analyzing...';
      default: return 'Unknown';
    }
  };

  if (isLoading) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.loadingContainer}>
          <Text style={styles.loadingText}>Loading your scans...</Text>
        </View>
      </SafeAreaView>
    );
  }

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
              My Scans ({scans.length})
            </Text>
            <Text style={[styles.subtitle, isLargeScreen && styles.subtitleLarge]}>
              Keep track of your skin health checks
              {isWeb ? ' - Enhanced desktop experience' : ''}
            </Text>
          </View>
          {isLargeScreen && (
            <View style={styles.platformIndicator}>
              {isWeb ? <Monitor size={24} color="#0066CC" /> : <Smartphone size={24} color="#0066CC" />}
            </View>
          )}
        </View>

        {/* Stats Dashboard */}
        <View style={[styles.statsContainer, isLargeScreen && styles.statsContainerLarge]}>
          <View style={[styles.statCard, isLargeScreen && styles.statCardLarge]}>
            <Eye size={isLargeScreen ? 28 : 24} color="#0066CC" strokeWidth={2} />
            <Text style={[styles.statValue, isLargeScreen && styles.statValueLarge]}>{stats.total}</Text>
            <Text style={[styles.statLabel, isLargeScreen && styles.statLabelLarge]}>Total Scans</Text>
          </View>
          <View style={[styles.statCard, isLargeScreen && styles.statCardLarge]}>
            <CheckCircle size={isLargeScreen ? 28 : 24} color="#10B981" strokeWidth={2} />
            <Text style={[styles.statValue, { color: '#10B981' }, isLargeScreen && styles.statValueLarge]}>{stats.benign}</Text>
            <Text style={[styles.statLabel, isLargeScreen && styles.statLabelLarge]}>Looks Good</Text>
          </View>
          <View style={[styles.statCard, isLargeScreen && styles.statCardLarge]}>
            <AlertTriangle size={isLargeScreen ? 28 : 24} color="#EF4444" strokeWidth={2} />
            <Text style={[styles.statValue, { color: '#EF4444' }, isLargeScreen && styles.statValueLarge]}>{stats.concerning}</Text>
            <Text style={[styles.statLabel, isLargeScreen && styles.statLabelLarge]}>Needs Attention</Text>
          </View>
          {stats.analyzing > 0 && (
            <View style={[styles.statCard, isLargeScreen && styles.statCardLarge]}>
              <Eye size={isLargeScreen ? 28 : 24} color="#F59E0B" strokeWidth={2} />
              <Text style={[styles.statValue, { color: '#F59E0B' }, isLargeScreen && styles.statValueLarge]}>{stats.analyzing}</Text>
              <Text style={[styles.statLabel, isLargeScreen && styles.statLabelLarge]}>Analyzing</Text>
            </View>
          )}
        </View>

        {/* Add New Scan Button */}
        <View style={[styles.actionsContainer, isLargeScreen && styles.actionsContainerLarge]}>
          <TouchableOpacity 
            style={[styles.actionButton, styles.primaryButton, isLargeScreen && styles.actionButtonLarge]} 
            onPress={addNewScan}>
            <Plus size={isLargeScreen ? 24 : 20} color="#FFFFFF" strokeWidth={2} />
            <Text style={[styles.actionButtonText, isLargeScreen && styles.actionButtonTextLarge]}>
              Add New Scan
            </Text>
          </TouchableOpacity>
        </View>

        {/* Filters */}
        <View style={[styles.filtersContainer, isLargeScreen && styles.filtersContainerLarge]}>
          <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.filters}>
            {(['all', 'benign', 'concerning'] as const).map((filterOption) => (
              <TouchableOpacity
                key={filterOption}
                style={[
                  styles.filterButton,
                  filter === filterOption && styles.filterButtonActive,
                  isLargeScreen && styles.filterButtonLarge
                ]}
                onPress={() => setFilter(filterOption)}>
                <Text style={[
                  styles.filterText,
                  filter === filterOption && styles.filterTextActive,
                  isLargeScreen && styles.filterTextLarge
                ]}>
                  {filterOption === 'all' ? 'All Scans' : 
                   filterOption === 'benign' ? 'Looks Good' : 'Needs Attention'}
                </Text>
              </TouchableOpacity>
            ))}
          </ScrollView>
        </View>

        {/* Health Reminder */}
        <View style={[styles.healthReminder, isLargeScreen && styles.healthReminderLarge]}>
          <Text style={[styles.reminderTitle, isLargeScreen && styles.reminderTitleLarge]}>
            💡 Remember
          </Text>
          <Text style={[styles.reminderText, isLargeScreen && styles.reminderTextLarge]}>
            This app helps you monitor your skin, but it's not a replacement for professional medical advice. 
            Always consult with a dermatologist for concerning results or changes in your skin.
          </Text>
        </View>

        {/* Scans Grid */}
        <View style={[styles.scansGrid, isLargeScreen && styles.scansGridLarge]}>
          {filteredScans.map((scan) => {
            const ResultIcon = getResultIcon(scan.result);
            const resultColor = getResultColor(scan.result);
            const isDeletingThisScan = isDeleting === scan.id;
            
            return (
              <View
                key={scan.id}
                style={[
                  styles.scanCard, 
                  isLargeScreen && styles.scanCardLarge,
                  isDeletingThisScan && styles.scanCardDeleting
                ]}>
                
                {/* Scan Image - Clickable to open details */}
                <TouchableOpacity
                  style={styles.scanImageContainer}
                  onPress={() => handleScanPress(scan)}
                  disabled={isDeletingThisScan}
                  activeOpacity={0.8}>
                  <Image source={{ uri: scan.uri }} style={[styles.scanPreview, isLargeScreen && styles.scanPreviewLarge]} />
                  {isDeletingThisScan && (
                    <View style={styles.deletingOverlay}>
                      <Text style={styles.deletingText}>Deleting...</Text>
                    </View>
                  )}
                </TouchableOpacity>
                
                {/* Scan Info */}
                <View style={[styles.scanInfo, isLargeScreen && styles.scanInfoLarge]}>
                  <View style={styles.resultContainer}>
                    <View style={[
                      styles.resultBadge,
                      { backgroundColor: `${resultColor}15` }
                    ]}>
                      <ResultIcon size={12} color={resultColor} strokeWidth={2} />
                      <Text style={[styles.resultText, { color: resultColor }, isLargeScreen && styles.resultTextLarge]}>
                        {getResultLabel(scan.result)}
                      </Text>
                    </View>
                    
                    {/* Delete Button - Separate from image touch area */}
                    <TouchableOpacity 
                      style={[
                        styles.deleteButton, 
                        isLargeScreen && styles.deleteButtonLarge,
                        isDeletingThisScan && styles.deleteButtonDisabled
                      ]}
                      onPress={(event) => handleDeleteScan(scan.id, event)}
                      disabled={isDeletingThisScan}
                      hitSlop={{ top: 25, bottom: 25, left: 25, right: 25 }}
                      activeOpacity={isDeletingThisScan ? 1 : 0.6}>
                      <Trash2 
                        size={isLargeScreen ? 20 : 18} 
                        color={isDeletingThisScan ? "#9CA3AF" : "#EF4444"} 
                        strokeWidth={2.5} 
                      />
                    </TouchableOpacity>
                  </View>
                  
                  {scan.confidence && (
                    <Text style={[styles.confidenceText, isLargeScreen && styles.confidenceTextLarge]}>
                      Confidence: {scan.confidence}%
                    </Text>
                  )}
                  
                  <Text style={[styles.dateText, isLargeScreen && styles.dateTextLarge]}>
                    {scan.dateScanned}
                  </Text>
                  
                  {scan.location && (
                    <Text style={[styles.locationText, isLargeScreen && styles.locationTextLarge]}>
                      {scan.location}
                    </Text>
                  )}
                </View>
              </View>
            );
          })}
        </View>

        {filteredScans.length === 0 && (
          <View style={[styles.emptyState, isLargeScreen && styles.emptyStateLarge]}>
            <Eye size={isLargeScreen ? 64 : 48} color="#9CA3AF" strokeWidth={2} />
            <Text style={[styles.emptyTitle, isLargeScreen && styles.emptyTitleLarge]}>
              No scans found
            </Text>
            <Text style={[styles.emptySubtitle, isLargeScreen && styles.emptySubtitleLarge]}>
              {filter === 'all' 
                ? 'Start by adding your first skin scan'
                : `No ${filter === 'benign' ? 'good results' : 'concerning results'} found`}
            </Text>
          </View>
        )}
      </ScrollView>

      {/* Details Modal */}
      <Modal
        visible={isViewingDetails}
        animationType="slide"
        presentationStyle="pageSheet">
        <ScanDetailsModal
          scan={selectedScan}
          onClose={() => {
            setIsViewingDetails(false);
            setSelectedScan(null);
          }}
          onDelete={handleDeleteScan}
          isLargeScreen={isLargeScreen}
          isDeleting={isDeleting}
        />
      </Modal>
    </SafeAreaView>
  );
}

interface ScanDetailsModalProps {
  scan: ScanResult | null;
  onClose: () => void;
  onDelete: (scanId: string, event?: any) => void;
  isLargeScreen: boolean;
  isDeleting: string | null;
}

function ScanDetailsModal({ scan, onClose, onDelete, isLargeScreen, isDeleting }: ScanDetailsModalProps) {
  if (!scan) return null;

  const getResultIcon = (result: string) => {
    switch (result) {
      case 'benign': return CheckCircle;
      case 'concerning': return AlertTriangle;
      case 'analyzing': return Eye;
      default: return CheckCircle;
    }
  };

  const getResultColor = (result: string) => {
    switch (result) {
      case 'benign': return '#10B981';
      case 'concerning': return '#EF4444';
      case 'analyzing': return '#F59E0B';
      default: return '#6B7280';
    }
  };

  const getResultLabel = (result: string) => {
    switch (result) {
      case 'benign': return 'Looks Good';
      case 'concerning': return 'See Doctor';
      case 'analyzing': return 'Analyzing...';
      default: return 'Unknown';
    }
  };

  const getResultDescription = (result: string) => {
    switch (result) {
      case 'benign': 
        return 'This scan shows characteristics typical of benign (non-cancerous) skin features. Continue regular monitoring and maintain good skin health practices.';
      case 'concerning': 
        return 'This scan shows some features that may need professional evaluation. We recommend scheduling an appointment with a dermatologist for a thorough examination.';
      case 'analyzing': 
        return 'Your scan is currently being analyzed. Results will be available shortly.';
      default: 
        return 'Unable to determine result.';
    }
  };

  const handleDelete = (event?: any) => {
    if (event) {
      event.stopPropagation();
      event.preventDefault();
    }
    onDelete(scan.id, event);
  };

  const ResultIcon = getResultIcon(scan.result);
  const resultColor = getResultColor(scan.result);
  const isDeletingThisScan = isDeleting === scan.id;

  return (
    <SafeAreaView style={styles.modalContainer}>
      <ScrollView style={styles.modalContent}>
        {/* Modal Header */}
        <View style={[styles.modalHeader, isLargeScreen && styles.modalHeaderLarge]}>
          <Text style={[styles.modalTitle, isLargeScreen && styles.modalTitleLarge]}>
            Scan Details
          </Text>
          <View style={styles.modalHeaderActions}>
            <TouchableOpacity 
              style={[
                styles.modalDeleteButton, 
                isLargeScreen && styles.modalDeleteButtonLarge,
                isDeletingThisScan && styles.modalDeleteButtonDisabled
              ]}
              onPress={handleDelete}
              disabled={isDeletingThisScan}
              hitSlop={{ top: 20, bottom: 20, left: 20, right: 20 }}
              activeOpacity={isDeletingThisScan ? 1 : 0.6}>
              <Trash2 
                size={isLargeScreen ? 22 : 20} 
                color={isDeletingThisScan ? "#9CA3AF" : "#EF4444"} 
                strokeWidth={2} 
              />
            </TouchableOpacity>
            <TouchableOpacity style={styles.closeButton} onPress={onClose}>
              <X size={isLargeScreen ? 28 : 24} color="#6B7280" strokeWidth={2} />
            </TouchableOpacity>
          </View>
        </View>

        {/* Image */}
        <View style={[styles.modalImageContainer, isLargeScreen && styles.modalImageContainerLarge]}>
          <Image 
            source={{ uri: scan.uri }} 
            style={[styles.modalImage, isLargeScreen && styles.modalImageLarge]} 
          />
          {isDeletingThisScan && (
            <View style={styles.modalDeletingOverlay}>
              <Text style={styles.modalDeletingText}>Deleting scan...</Text>
            </View>
          )}
        </View>

        {/* Result */}
        <View style={[styles.resultSection, isLargeScreen && styles.resultSectionLarge]}>
          <View style={styles.resultHeader}>
            <ResultIcon size={isLargeScreen ? 32 : 24} color={resultColor} strokeWidth={2} />
            <Text style={[styles.resultTitle, { color: resultColor }, isLargeScreen && styles.resultTitleLarge]}>
              {getResultLabel(scan.result)}
            </Text>
          </View>
          
          <Text style={[styles.resultDescription, isLargeScreen && styles.resultDescriptionLarge]}>
            {getResultDescription(scan.result)}
          </Text>

          {scan.confidence && (
            <View style={[styles.confidenceContainer, isLargeScreen && styles.confidenceContainerLarge]}>
              <Text style={[styles.confidenceLabel, isLargeScreen && styles.confidenceLabelLarge]}>
                Analysis Confidence
              </Text>
              <Text style={[styles.confidenceValue, isLargeScreen && styles.confidenceValueLarge]}>
                {scan.confidence}%
              </Text>
            </View>
          )}
        </View>

        {/* Scan Information */}
        <View style={[styles.infoSection, isLargeScreen && styles.infoSectionLarge]}>
          <Text style={[styles.sectionTitle, isLargeScreen && styles.sectionTitleLarge]}>
            Scan Information
          </Text>
          
          <View style={[styles.infoRow, isLargeScreen && styles.infoRowLarge]}>
            <Text style={[styles.infoLabel, isLargeScreen && styles.infoLabelLarge]}>Date Scanned</Text>
            <Text style={[styles.infoValue, isLargeScreen && styles.infoValueLarge]}>{scan.dateScanned}</Text>
          </View>

          {scan.location && (
            <View style={[styles.infoRow, isLargeScreen && styles.infoRowLarge]}>
              <Text style={[styles.infoLabel, isLargeScreen && styles.infoLabelLarge]}>Location</Text>
              <Text style={[styles.infoValue, isLargeScreen && styles.infoValueLarge]}>{scan.location}</Text>
            </View>
          )}
        </View>

        {/* Recommendations */}
        {scan.result === 'concerning' && (
          <View style={[styles.recommendationSection, isLargeScreen && styles.recommendationSectionLarge]}>
            <Text style={[styles.recommendationTitle, isLargeScreen && styles.recommendationTitleLarge]}>
              Next Steps
            </Text>
            <Text style={[styles.recommendationText, isLargeScreen && styles.recommendationTextLarge]}>
              • Schedule an appointment with a dermatologist{'\n'}
              • Monitor the area for any changes{'\n'}
              • Take photos to track changes over time{'\n'}
              • Avoid excessive sun exposure to the area
            </Text>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
  },
  largeScreenContainer: {
    maxWidth: 1400,
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
  statsContainer: {
    flexDirection: 'row',
    paddingHorizontal: 24,
    marginBottom: 16,
    gap: 12,
  },
  statsContainerLarge: {
    paddingHorizontal: 40,
    gap: 20,
  },
  statCard: {
    flex: 1,
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  statCardLarge: {
    padding: 24,
    borderRadius: 16,
  },
  statValue: {
    fontSize: 24,
    fontFamily: 'Inter-Bold',
    color: '#111827',
    marginTop: 8,
    marginBottom: 4,
  },
  statValueLarge: {
    fontSize: 32,
    marginTop: 12,
  },
  statLabel: {
    fontSize: 12,
    fontFamily: 'Inter-Medium',
    color: '#6B7280',
    textAlign: 'center',
  },
  statLabelLarge: {
    fontSize: 14,
  },
  actionsContainer: {
    paddingHorizontal: 24,
    marginBottom: 16,
  },
  actionsContainerLarge: {
    paddingHorizontal: 40,
  },
  actionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
    gap: 8,
  },
  actionButtonLarge: {
    paddingVertical: 16,
    borderRadius: 12,
    gap: 12,
  },
  primaryButton: {
    backgroundColor: '#0066CC',
  },
  actionButtonText: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#FFFFFF',
  },
  actionButtonTextLarge: {
    fontSize: 18,
  },
  healthReminder: {
    margin: 24,
    marginTop: 0,
    padding: 16,
    backgroundColor: '#EBF4FF',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#BFDBFE',
  },
  healthReminderLarge: {
    margin: 40,
    marginTop: 0,
    padding: 20,
    borderRadius: 16,
  },
  reminderTitle: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#1E40AF',
    marginBottom: 8,
  },
  reminderTitleLarge: {
    fontSize: 18,
    marginBottom: 10,
  },
  reminderText: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#1E3A8A',
    lineHeight: 20,
  },
  reminderTextLarge: {
    fontSize: 16,
    lineHeight: 24,
  },
  filtersContainer: {
    paddingHorizontal: 24,
    marginBottom: 16,
  },
  filtersContainerLarge: {
    paddingHorizontal: 40,
  },
  filters: {
    flexDirection: 'row',
    gap: 12,
  },
  filterButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#E5E7EB',
  },
  filterButtonLarge: {
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 24,
  },
  filterButtonActive: {
    backgroundColor: '#0066CC',
  },
  filterText: {
    fontSize: 14,
    fontFamily: 'Inter-Medium',
    color: '#6B7280',
  },
  filterTextLarge: {
    fontSize: 16,
  },
  filterTextActive: {
    color: '#FFFFFF',
  },
  scansGrid: {
    paddingHorizontal: 24,
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
  },
  scansGridLarge: {
    paddingHorizontal: 40,
    gap: 24,
  },
  scanCard: {
    width: (width - 64) / 2,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    overflow: 'hidden',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 8,
    elevation: 2,
  },
  scanCardLarge: {
    width: (width - 128) / 3,
    borderRadius: 16,
  },
  scanCardDeleting: {
    opacity: 0.6,
  },
  scanImageContainer: {
    width: '100%',
    position: 'relative',
  },
  scanPreview: {
    width: '100%',
    height: 120,
  },
  scanPreviewLarge: {
    height: 160,
  },
  deletingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  deletingText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontFamily: 'Inter-SemiBold',
  },
  scanInfo: {
    padding: 12,
  },
  scanInfoLarge: {
    padding: 16,
  },
  resultContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  resultBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
    flex: 1,
    marginRight: 8,
  },
  resultText: {
    fontSize: 12,
    fontFamily: 'Inter-SemiBold',
  },
  resultTextLarge: {
    fontSize: 14,
  },
  deleteButton: {
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#FEF2F2',
    borderWidth: 2,
    borderColor: '#FECACA',
    minWidth: 44,
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#EF4444',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 3,
    // Ensure button is above other elements
    zIndex: 10,
  },
  deleteButtonLarge: {
    padding: 14,
    borderRadius: 14,
    minWidth: 48,
    minHeight: 48,
  },
  deleteButtonDisabled: {
    backgroundColor: '#F3F4F6',
    borderColor: '#E5E7EB',
    shadowOpacity: 0,
    elevation: 0,
  },
  confidenceText: {
    fontSize: 11,
    fontFamily: 'Inter-Medium',
    color: '#6B7280',
    marginBottom: 4,
  },
  confidenceTextLarge: {
    fontSize: 12,
  },
  dateText: {
    fontSize: 10,
    fontFamily: 'Inter-Regular',
    color: '#9CA3AF',
    marginBottom: 2,
  },
  dateTextLarge: {
    fontSize: 11,
  },
  locationText: {
    fontSize: 10,
    fontFamily: 'Inter-Regular',
    color: '#9CA3AF',
  },
  locationTextLarge: {
    fontSize: 11,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: 60,
    paddingHorizontal: 24,
  },
  emptyStateLarge: {
    paddingVertical: 80,
    paddingHorizontal: 40,
  },
  emptyTitle: {
    fontSize: 20,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginTop: 16,
    marginBottom: 8,
  },
  emptyTitleLarge: {
    fontSize: 28,
    marginTop: 24,
    marginBottom: 12,
  },
  emptySubtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#6B7280',
    textAlign: 'center',
  },
  emptySubtitleLarge: {
    fontSize: 18,
  },
  // Modal Styles
  modalContainer: {
    flex: 1,
    backgroundColor: '#F9FAFB',
  },
  modalContent: {
    flex: 1,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 24,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#E5E7EB',
  },
  modalHeaderLarge: {
    padding: 32,
    paddingBottom: 24,
  },
  modalTitle: {
    fontSize: 24,
    fontFamily: 'Inter-Bold',
    color: '#111827',
  },
  modalTitleLarge: {
    fontSize: 32,
  },
  modalHeaderActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  modalDeleteButton: {
    padding: 12,
    borderRadius: 12,
    backgroundColor: '#FEF2F2',
    borderWidth: 2,
    borderColor: '#FECACA',
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#EF4444',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 4,
    elevation: 3,
  },
  modalDeleteButtonLarge: {
    padding: 14,
    borderRadius: 14,
  },
  modalDeleteButtonDisabled: {
    backgroundColor: '#F3F4F6',
    borderColor: '#E5E7EB',
    shadowOpacity: 0,
    elevation: 0,
  },
  closeButton: {
    padding: 8,
  },
  modalImageContainer: {
    alignItems: 'center',
    padding: 24,
    position: 'relative',
  },
  modalImageContainerLarge: {
    padding: 32,
  },
  modalImage: {
    width: 250,
    height: 250,
    borderRadius: 16,
  },
  modalImageLarge: {
    width: 350,
    height: 350,
    borderRadius: 20,
  },
  modalDeletingOverlay: {
    position: 'absolute',
    top: 24,
    left: 24,
    right: 24,
    bottom: 24,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalDeletingText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontFamily: 'Inter-SemiBold',
  },
  resultSection: {
    padding: 24,
    paddingTop: 16,
  },
  resultSectionLarge: {
    padding: 32,
    paddingTop: 24,
  },
  resultHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 12,
  },
  resultTitle: {
    fontSize: 24,
    fontFamily: 'Inter-Bold',
  },
  resultTitleLarge: {
    fontSize: 32,
  },
  resultDescription: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: '#374151',
    lineHeight: 24,
    marginBottom: 16,
  },
  resultDescriptionLarge: {
    fontSize: 18,
    lineHeight: 28,
    marginBottom: 20,
  },
  confidenceContainer: {
    backgroundColor: '#F3F4F6',
    padding: 16,
    borderRadius: 12,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  confidenceContainerLarge: {
    padding: 20,
    borderRadius: 16,
  },
  confidenceLabel: {
    fontSize: 14,
    fontFamily: 'Inter-Medium',
    color: '#6B7280',
  },
  confidenceLabelLarge: {
    fontSize: 16,
  },
  confidenceValue: {
    fontSize: 18,
    fontFamily: 'Inter-Bold',
    color: '#111827',
  },
  confidenceValueLarge: {
    fontSize: 22,
  },
  infoSection: {
    padding: 24,
    paddingTop: 16,
  },
  infoSectionLarge: {
    padding: 32,
    paddingTop: 24,
  },
  sectionTitle: {
    fontSize: 18,
    fontFamily: 'Inter-SemiBold',
    color: '#111827',
    marginBottom: 16,
  },
  sectionTitleLarge: {
    fontSize: 22,
    marginBottom: 20,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  infoRowLarge: {
    paddingVertical: 16,
  },
  infoLabel: {
    fontSize: 14,
    fontFamily: 'Inter-Medium',
    color: '#6B7280',
  },
  infoLabelLarge: {
    fontSize: 16,
  },
  infoValue: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#111827',
  },
  infoValueLarge: {
    fontSize: 16,
  },
  recommendationSection: {
    margin: 24,
    padding: 20,
    backgroundColor: '#FEF2F2',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#FECACA',
  },
  recommendationSectionLarge: {
    margin: 32,
    padding: 24,
    borderRadius: 16,
  },
  recommendationTitle: {
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    color: '#DC2626',
    marginBottom: 12,
  },
  recommendationTitleLarge: {
    fontSize: 18,
    marginBottom: 16,
  },
  recommendationText: {
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    color: '#7F1D1D',
    lineHeight: 20,
  },
  recommendationTextLarge: {
    fontSize: 16,
    lineHeight: 24,
  },
});