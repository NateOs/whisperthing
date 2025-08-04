-- SQL Server database setup script
-- Run this script to create the database and tables

-- Create database
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'CallAnalysis')
BEGIN
    CREATE DATABASE CallAnalysis;
END
GO

USE CallAnalysis;
GO

-- Create calls table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='calls' AND xtype='U')
BEGIN
    CREATE TABLE calls (
        id INT IDENTITY(1,1) PRIMARY KEY,
        audio_file_path NVARCHAR(500) NOT NULL,
        processed_date DATETIME2 DEFAULT GETDATE(),
        duration_seconds FLOAT,
        file_size_bytes BIGINT,
        status NVARCHAR(50) DEFAULT 'completed',
        INDEX IX_calls_processed_date (processed_date),
        INDEX IX_calls_status (status)
    );
END
GO

-- Create transcriptions table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='transcriptions' AND xtype='U')
BEGIN
    CREATE TABLE transcriptions (
        id INT IDENTITY(1,1) PRIMARY KEY,
        call_id INT NOT NULL,
        speaker_type NVARCHAR(20) NOT NULL, -- 'agent' or 'customer'
        text NVARCHAR(MAX) NOT NULL,
        start_time FLOAT NOT NULL,
        end_time FLOAT NOT NULL,
        confidence FLOAT,
        FOREIGN KEY (call_id) REFERENCES calls(id) ON DELETE CASCADE,
        INDEX IX_transcriptions_call_id (call_id),
        INDEX IX_transcriptions_speaker (speaker_type),
        INDEX IX_transcriptions_time (start_time, end_time)
    );
END
GO

-- Create keyword_detections table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='keyword_detections' AND xtype='U')
BEGIN
    CREATE TABLE keyword_detections (
        id INT IDENTITY(1,1) PRIMARY KEY,
        call_id INT NOT NULL,
        keyword NVARCHAR(100) NOT NULL,
        speaker_type NVARCHAR(20) NOT NULL,
        timestamp_seconds FLOAT NOT NULL,
        context_text NVARCHAR(500),
        confidence FLOAT,
        FOREIGN KEY (call_id) REFERENCES calls(id) ON DELETE CASCADE,
        INDEX IX_keyword_detections_call_id (call_id),
        INDEX IX_keyword_detections_keyword (keyword),
        INDEX IX_keyword_detections_speaker (speaker_type),
        INDEX IX_keyword_detections_timestamp (timestamp_seconds)
    );
END
GO

-- Create call_metrics table
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='call_metrics' AND xtype='U')
BEGIN
    CREATE TABLE call_metrics (
        id INT IDENTITY(1,1) PRIMARY KEY,
        call_id INT NOT NULL,
        total_agent_talk_time FLOAT,
        total_customer_talk_time FLOAT,
        agent_word_count INT,
        customer_word_count INT,
        total_keywords_found INT,
        politeness_score FLOAT,
        FOREIGN KEY (call_id) REFERENCES calls(id) ON DELETE CASCADE,
        INDEX IX_call_metrics_call_id (call_id)
    );
END
GO

-- Create a view for easy reporting
CREATE OR ALTER VIEW vw_call_summary AS
SELECT 
    c.id,
    c.audio_file_path,
    c.processed_date,
    c.duration_seconds,
    c.file_size_bytes,
    cm.total_agent_talk_time,
    cm.total_customer_talk_time,
    cm.agent_word_count,
    cm.customer_word_count,
    cm.total_keywords_found,
    cm.politeness_score,
    (SELECT COUNT(*) FROM keyword_detections kd WHERE kd.call_id = c.id AND kd.speaker_type = 'agent') as agent_keywords,
    (SELECT COUNT(*) FROM keyword_detections kd WHERE kd.call_id = c.id AND kd.speaker_type = 'customer') as customer_keywords
FROM calls c
LEFT JOIN call_metrics cm ON c.id = cm.call_id;
GO

PRINT 'Database setup completed successfully!';